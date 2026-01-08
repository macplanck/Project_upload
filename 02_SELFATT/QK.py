from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass
class QK:
    """
    QK block simulator.

    Model parameters
    ----------------
    hiddenSize : int
        Total hidden size (e.g., 4096).
    headNum : int
        Number of heads (e.g., 32).
    seqLen : int
        Sequence length (e.g., 8192).

    Control signals (state)
    -----------------------
    rowCnt : int
        Which row of K we are evaluating. Range: [0, seqLen-1]
    colCnt : int
        Which column-block within (hiddenSize/headNum) we are working on.
        Range: [0, (hiddenSize/headNum)-1]
    qFlag : int
        1 means current input is Q; 0 means current input is K.

    Buffers and storage
    -------------------
    q_deq : np.ndarray
        Shape (32,), dtype float16. Dequantized Q chunk.
    k_deq : np.ndarray
        Shape (32,), dtype float16. Dequantized K chunk.
    accu : np.ndarray
        Shape (hiddenSize/headNum,), dtype float16. Accumulator registers.

    Cycle counter
    -------------
    trgCnt : int
        Increments each call to one_operation().

    Latency parameter
    -----------------
    lat : int
        Default hardware latency. Stored for reference; not pipelined unless adding it.
    """

    # ---- model parameters ----
    hiddenSize: int = 4096
    headNum: int = 32
    seqLen: int = 8192

    # ---- default latency ----
    lat: int = 4

    # ---- control signals ----
    rowCnt: int = 0
    colCnt: int = 0
    qFlag: int = 1  # default assumption: start by loading Q

    # ---- cycle counter ----
    trgCnt: int = 0

    # ---- internal storage ----
    q_deq: np.ndarray = None
    k_deq: np.ndarray = None
    accu: np.ndarray = None

    def __post_init__(self) -> None:
        if self.hiddenSize % self.headNum != 0:
            raise ValueError("hiddenSize must be divisible by headNum.")
        self.cols_per_head = self.hiddenSize // self.headNum  # e.g., 128

        self.q_deq = np.zeros((32,), dtype=np.float16)
        self.k_deq = np.zeros((32,), dtype=np.float16)
        self.accu = np.zeros((self.cols_per_head,), dtype=np.float16)

        # sanitize initial control ranges
        self.rowCnt %= self.seqLen
        self.colCnt %= self.cols_per_head
        self.qFlag = 1 if self.qFlag else 0

    def one_operation(
        self,
        inNum: Sequence[int],
        dqFac: int,
        q_flag: Optional[int] = None,
    ) -> np.float16:
        """
        Perform one cycle operation.

        Parameters
        ----------
        inNum : Sequence[int]
            32 x int8 input values (quantized).
        dqFac : int
            5-bit integer dequant shift. Interpreted as division by 2**dqFac.
        q_flag : Optional[int]
            If provided, overrides internal qFlag for *this* operation only.
            If None, uses self.qFlag.

        Returns
        -------
        np.float16
            The updated partial sum written into accu[colCnt].
        """
        if q_flag is None:
            q_flag_eff = self.qFlag
        else:
            q_flag_eff = 1 if q_flag else 0

        # Compute and write back
        new_sum = self.eval(inNum=inNum, dqFac=dqFac, q_flag=q_flag_eff)
        self.accu[self.colCnt] = new_sum

        # Update control and cycle count
        self.update_ctrl(q_flag=q_flag_eff)
        self.trgCnt += 1

        return new_sum

    def eval(self, *, inNum: Sequence[int], dqFac: int, q_flag: int) -> np.float16:
        """
        Evaluate accu[colCnt] update given current input.

        Pseudocode mapping
        ------------------
        deq = float16(inNum / 2**dqFac)
        if q_flag:
            q_deq = deq
            sum = accu[colCnt]
        else:
            k_deq = deq
            sum = sum_i(k_deq[i] * q_deq[i]) + accu[colCnt]
        return sum
        """
        in_arr = np.asarray(inNum, dtype=np.int8)
        if in_arr.shape != (32,):
            raise ValueError(f"inNum must have shape (32,), got {in_arr.shape}.")

        if not (0 <= int(dqFac) <= 31):
            raise ValueError("dqFac must be a 5-bit integer in [0, 31].")

        scale = 2.0 ** int(dqFac)
        deq = (in_arr.astype(np.float16) / scale).astype(np.float16)

        if q_flag:
            self.q_deq = deq
            # In Q-load cycle, output is just the current accumulator value.
            out = np.float16(self.accu[self.colCnt])
        else:
            self.k_deq = deq
            # Dot product in float32, then add previous accu[colCnt]
            dot = np.sum(self.k_deq.astype(np.float16) * self.q_deq.astype(np.float16))
            out = dot + np.float16(self.accu[self.colCnt])

        # Store back as FP16 register behavior
        return np.float16(out)

    def update_ctrl(self, *, q_flag: int) -> None:
        """
        Update control signals according to pseudocode.

        Pseudocode mapping
        ------------------
        if colCnt == (cols_per_head - 1):
            if rowCnt == (seqLen - 1):
                rowCnt = 0
                colCnt = 0
                qFlag = 1
                accu[:] = 0
            else:
                rowCnt += 1
                colCnt = 0
        elif (~q_flag):
            colCnt += 1
        elif (q_flag):
            qFlag = 0
        """
        last_col = (self.colCnt == (self.cols_per_head - 1))
        last_row = (self.rowCnt == (self.seqLen - 1))

        if last_col:
            if last_row:
                self.rowCnt = 0
                self.colCnt = 0
                self.qFlag = 1
                self.accu[:] = np.float16(0.0)
            else:
                self.rowCnt += 1
                self.colCnt = 0
                # Keep qFlag as-is unless design wants to reload Q each row.
        elif q_flag == 0:
            self.colCnt += 1
        else:
            # q_flag == 1: after loading Q once, switch to K phase
            self.qFlag = 0

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== QK Sanity Test ===")

    # Instantiate QK
    qk = QK()

    # -----------------------------
    # Test vectors
    # -----------------------------
    Q_in = np.ones(32, dtype=np.int8)        # Q = [1, 1, ..., 1]
    K_in = np.full(32, 2, dtype=np.int8)     # K = [2, 2, ..., 2]
    dqFac = 0                                # no scaling

    # -----------------------------
    # Cycle 0: Q phase
    # -----------------------------
    print("\n[Cycle 0] Q phase")
    out0 = qk.one_operation(Q_in, dqFac)

    print(f"out        = {out0}")
    print(f"qFlag      = {qk.qFlag}")
    print(f"rowCnt     = {qk.rowCnt}")
    print(f"colCnt     = {qk.colCnt}")
    print(f"trgCnt     = {qk.trgCnt}")

    # Checks
    assert out0 == 0.0, "Accumulator should not change during Q phase"
    assert qk.qFlag == 0, "qFlag should flip low after Q load"
    assert qk.colCnt == 0, "colCnt should not advance during Q phase"

    # -----------------------------
    # Cycle 1: K phase
    # -----------------------------
    print("\n[Cycle 1] K phase")
    out1 = qk.one_operation(K_in, dqFac)

    expected_dot = 32 * 2  # sum(Q * K)
    print(f"out        = {out1}")
    print(f"expected   = {expected_dot}")
    print(f"qFlag      = {qk.qFlag}")
    print(f"rowCnt     = {qk.rowCnt}")
    print(f"colCnt     = {qk.colCnt}")
    print(f"trgCnt     = {qk.trgCnt}")

    # Checks
    assert np.isclose(out1, expected_dot), "Dot product incorrect"
    assert qk.colCnt == 1, "colCnt should increment during K phase"
    assert qk.trgCnt == 2, "trgCnt should increment every cycle"

    # -----------------------------
    # Additional K cycles
    # -----------------------------
    print("\n[Additional K cycles]")
    for i in range(2, 6):
        out = qk.one_operation(K_in, dqFac)
        print(f"Cycle {i}: out={out}, colCnt={qk.colCnt}")

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n=== Test PASSED ===")
