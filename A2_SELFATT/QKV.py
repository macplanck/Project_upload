from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple
import numpy as np


class SpecViolationError(AssertionError):
    """Raised when the implementation detects a specification violation."""


def fp16(x: float | np.floating | np.ndarray) -> np.float16 | np.ndarray:
    """
    Cast scalar/array to FP16 with IEEE-754 binary16 rounding.

    Notes
    -----
    This is the primary mechanism used here to enforce the rule:
    'all floating point operation should be conducted in float16'.
    """
    return np.asarray(x, dtype=np.float16)


def adder_tree_fp16(vec32: np.ndarray) -> np.float16:
    """
    Reduce 32 FP16 numbers using an FP16 pairwise adder tree.

    Parameters
    ----------
    vec32 : np.ndarray
        Shape (32,), dtype float16 (or castable).

    Returns
    -------
    np.float16
        FP16-reduced sum with FP16 rounding after every add.

    Raises
    ------
    ValueError
        If input shape is not (32,).
    """
    x = np.asarray(vec32, dtype=np.float16)
    if x.shape != (32,):
        raise ValueError(f"adder_tree_fp16 expects shape (32,), got {x.shape}")

    # Pairwise tree: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    # FP16 rounding after each add by casting back to np.float16.
    stage = x
    while stage.shape[0] > 1:
        stage = fp16(stage[0::2] + stage[1::2])  # FP16 add then quantize
    return np.float16(stage[0])


@dataclass
class QKV:
    """
    QKV block simulator: computes one row of V * softmax(QK^T) in a streaming schedule.

    Model parameters
    ----------------
    hiddenSize : int
        Total hidden size (default 4096).
    headNum : int
        Number of heads (default 32).
    seqLen : int
        Sequence length (default 8192).
    lat : int
        Nominal hardware latency annotation (default 3). Not modeled as a delay here.

    Control/state
    -------------
    sliceCnt : int
        Which 32-token slice of the softmax row we are processing. Range: [0, seqLen/32 - 1]
    colCnt : int
        Which head-dim index (d) we are updating. Range: [0, Dh - 1]
    QKFlag : bool
        True means the current call provides QK slice (softmax weights) to load QKbuf.
        False means the current call provides V slice to compute dot and accumulate.

    Storage
    -------
    QKbuf : np.ndarray
        Shape (32,), dtype float16.
    Vbuf : np.ndarray
        Shape (32,), dtype float16.
    accu : np.ndarray
        Shape (Dh,), dtype float16. Accumulator across slices.

    Spec checker
    ------------
    Enforces: per slice => exactly 1 QK call followed by Dh V calls, in that order.
    """

    hiddenSize: int = 4096
    headNum: int = 32
    seqLen: int = 8192
    lat: int = 3

    sliceCnt: int = 0
    colCnt: int = 0
    QKFlag: bool = True

    trgCnt: int = 0

    QKbuf: np.ndarray = field(default_factory=lambda: np.zeros((32,), dtype=np.float16))
    Vbuf: np.ndarray = field(default_factory=lambda: np.zeros((32,), dtype=np.float16))
    accu: np.ndarray = field(init=False)

    # Internal checker state
    _expectedPhase: str = field(default="QK", init=False)  # "QK" then "V"
    _vCallsRemainingInSlice: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.seqLen % 32 != 0:
            raise ValueError(f"seqLen must be divisible by 32, got seqLen={self.seqLen}")
        if self.hiddenSize % self.headNum != 0:
            raise ValueError(
                f"hiddenSize must be divisible by headNum, got {self.hiddenSize}/{self.headNum}"
            )

        dh = self.dh
        self.accu = np.zeros((dh,), dtype=np.float16)

        # For a fresh slice, we expect 1 QK call then Dh V calls.
        self._expectedPhase = "QK"
        self._vCallsRemainingInSlice = dh

        # Basic sanity of counters
        self._assert_ctrl_ranges()

    @property
    def dh(self) -> int:
        """Head dimension (hiddenSize/headNum)."""
        return self.hiddenSize // self.headNum

    @property
    def sliceMax(self) -> int:
        """Maximum sliceCnt value."""
        return (self.seqLen // 32) - 1

    def _assert_ctrl_ranges(self) -> None:
        if not (0 <= self.sliceCnt <= self.sliceMax):
            raise SpecViolationError(f"sliceCnt out of range: {self.sliceCnt}")
        if not (0 <= self.colCnt <= self.dh - 1):
            raise SpecViolationError(f"colCnt out of range: {self.colCnt}")

    def _checker_before_call(self) -> None:
        """
        Assert the per-slice sequence rule:
        - Exactly 1 QK call then Dh V calls.
        """
        if self._expectedPhase == "QK":
            if not self.QKFlag:
                raise SpecViolationError(
                    "Spec violation: expected QKFlag==True for the QK-load call of the slice."
                )
            if self._vCallsRemainingInSlice != self.dh:
                raise SpecViolationError(
                    "Internal checker inconsistency: vCallsRemainingInSlice should equal Dh at slice start."
                )

        elif self._expectedPhase == "V":
            if self.QKFlag:
                raise SpecViolationError(
                    "Spec violation: expected QKFlag==False during V-compute calls."
                )
            if self._vCallsRemainingInSlice <= 0:
                raise SpecViolationError(
                    "Spec violation: too many V calls in a slice (exceeded Dh)."
                )
        else:
            raise SpecViolationError(f"Internal checker: unknown phase {self._expectedPhase!r}")

    def _checker_after_call(self) -> None:
        """
        Update checker expectations after a successful call.
        """
        if self._expectedPhase == "QK":
            # After QK load, we must see Dh V calls
            self._expectedPhase = "V"
            self._vCallsRemainingInSlice = self.dh
        else:
            # Consumed one V call
            self._vCallsRemainingInSlice -= 1
            if self._vCallsRemainingInSlice == 0:
                # Next call starts a new slice => expect QK
                self._expectedPhase = "QK"
                self._vCallsRemainingInSlice = self.dh

    def one_operation(self, inNum: Iterable[np.float16] | np.ndarray) -> np.float16:
        """
        Execute one cycle-equivalent operation.

        Parameters
        ----------
        inNum : iterable/np.ndarray
            32 FP16 numbers.

        Returns
        -------
        np.float16
            prtSum (partial sum). Note: it is the FINAL output element only when:
              - QKFlag == False (V-compute cycle), and
              - sliceCnt == sliceMax (last slice of the row).
            The memory controller is assumed to read QKFlag/sliceCnt/colCnt as control signals.
        """
        self._assert_ctrl_ranges()
        self._checker_before_call()

        prtSum = self.eval(inNum)

        # Spec hygiene: only mutate accu on V-compute cycles.
        if not self.QKFlag:
            self.accu[self.colCnt] = np.float16(prtSum)

        self.update_ctrl()
        self.trgCnt += 1

        self._checker_after_call()
        return np.float16(prtSum)

    def eval(self, inNum: Iterable[np.float16] | np.ndarray) -> np.float16:
        """
        Evaluate the datapath for the current cycle.

        - If QKFlag: load QKbuf, return current accu[colCnt] (logically consistent no-op).
        - Else: load Vbuf, compute dot32(QKbuf, Vbuf) with FP16 adder tree,
          then add to accu[colCnt] in FP16.
        """
        vec = np.asarray(list(inNum), dtype=np.float16)
        if vec.shape != (32,):
            raise SpecViolationError(f"inNum must be 32 elements, got shape {vec.shape}")

        if self.QKFlag:
            self.QKbuf = vec
            partialSum = np.float16(self.accu[self.colCnt])
            return np.float16(partialSum)

        self.Vbuf = vec

        # Multiply lane-wise in FP16
        prod = fp16(self.Vbuf * self.QKbuf)  # elementwise multiply then quantize to FP16

        # Reduce using FP16 adder tree
        dot = adder_tree_fp16(prod)

        # Accumulate into accu[colCnt] in FP16 (quantize the add)
        partialSum = fp16(np.float16(dot) + np.float16(self.accu[self.colCnt]))
        return np.float16(partialSum)

    def update_ctrl(self) -> None:
        """
        Control update per spec pseudocode.
        """
        lastCol = (self.dh - 1)
        lastSlice = self.sliceMax

        if self.colCnt == lastCol:
            self.QKFlag = True
            self.colCnt = 0

            if self.sliceCnt == lastSlice:
                self.sliceCnt = 0
                self.accu[:] = np.float16(0.0)
            else:
                self.sliceCnt += 1

        elif not self.QKFlag:
            self.colCnt += 1

        else:
            # QKFlag is True and we are not at last column => next cycle is V compute
            self.QKFlag = False

        self._assert_ctrl_ranges()

    # Convenience helpers for external logic / testing
    def is_v_cycle(self) -> bool:
        """True if current cycle is a V-compute cycle (QKFlag==False)."""
        return not self.QKFlag

    def is_final_output_cycle(self) -> bool:
        """
        True if the returned value corresponds to the final output element for the row:
        (V-compute cycle) AND (sliceCnt == lastSlice).
        """
        return (not self.QKFlag) and (self.sliceCnt == self.sliceMax)


####################################################
#######         verification helpers          ######
####################################################
def golden_row_v_softmax_tree_blocked(
    softmax_row: np.ndarray, v_mat: np.ndarray
) -> np.ndarray:
    """
    Golden computation that matches QKV spec:

      For each slice (32 tokens):
        dot32(d) = adder_tree_fp16( fp16(Vslice[:,d] * QKslice[:]) )
        accu[d]  = fp16(accu[d] + dot32(d))

    Parameters
    ----------
    softmax_row : np.ndarray
        Shape (seqLen,), dtype float16.
    v_mat : np.ndarray
        Shape (seqLen, Dh), dtype float16.

    Returns
    -------
    np.ndarray
        Shape (Dh,), dtype float16.
    """
    soft = np.asarray(softmax_row, dtype=np.float16)
    v = np.asarray(v_mat, dtype=np.float16)

    if soft.ndim != 1:
        raise ValueError("softmax_row must be 1-D")
    if v.ndim != 2:
        raise ValueError("v_mat must be 2-D")
    if v.shape[0] != soft.shape[0]:
        raise ValueError("seqLen mismatch between softmax_row and v_mat")
    if soft.shape[0] % 32 != 0:
        raise ValueError("seqLen must be divisible by 32")

    seqLen = soft.shape[0]
    dh = v.shape[1]
    slices = seqLen // 32

    out = np.zeros((dh,), dtype=np.float16)

    for s in range(slices):
        qk = soft[s * 32 : (s + 1) * 32]  # (32,)
        for d in range(dh):
            vs = v[s * 32 : (s + 1) * 32, d]  # (32,)
            prod = fp16(vs * qk)              # FP16 multiply + quantize
            dot = adder_tree_fp16(prod)       # FP16 tree reduce
            out[d] = np.float16(fp16(np.float16(out[d]) + np.float16(dot)))  # FP16 accumulate
    return out


def run_one_row(qkv: QKV, softmax_row: np.ndarray, v_mat: np.ndarray) -> np.ndarray:
    """
    Drive QKV for exactly one attention output row and return the final Dh outputs.

    Assumes qkv begins at a row boundary:
      QKFlag==True, sliceCnt==0, colCnt==0
    """
    Dh = qkv.dh
    seqLen = qkv.seqLen
    slices = seqLen // 32

    if not (qkv.QKFlag and qkv.sliceCnt == 0 and qkv.colCnt == 0):
        raise SpecViolationError(
            f"Driver expects row-start state, got QKFlag={qkv.QKFlag}, "
            f"sliceCnt={qkv.sliceCnt}, colCnt={qkv.colCnt}"
        )

    captured_final = np.zeros((Dh,), dtype=np.float16)

    for sliceCnt in range(slices):
        # 1x QK load
        qk_slice = softmax_row[sliceCnt * 32 : (sliceCnt + 1) * 32]
        _ = qkv.one_operation(qk_slice)

        # Dh x V compute
        for d in range(Dh):
            v_slice = v_mat[sliceCnt * 32 : (sliceCnt + 1) * 32, d]
            val = qkv.one_operation(v_slice)

            if sliceCnt == (slices - 1):
                captured_final[d] = np.float16(val)

    if not (qkv.QKFlag and qkv.sliceCnt == 0 and qkv.colCnt == 0):
        raise SpecViolationError(
            "Spec violation: expected row reset after finishing a row "
            f"(QKFlag={qkv.QKFlag}, sliceCnt={qkv.sliceCnt}, colCnt={qkv.colCnt})"
        )

    return captured_final


def mixed_err(a: np.ndarray, b: np.ndarray) -> float:
    """
    Robust scalar error metric:
      max_abs_diff / max(1e-3, max_abs(b))
    Avoids blowing up when b is near zero.
    """
    diff = a.astype(np.float32) - b.astype(np.float32)
    abs_err = float(np.max(np.abs(diff)))
    scale = float(max(1e-3, np.max(np.abs(b.astype(np.float32)))))
    return abs_err / scale


def make_softmax_row_fp16(rng: np.random.Generator, seqLen: int) -> np.ndarray:
    """
    Generate a random "softmax-like" row (nonnegative, sums to 1) in FP16-ish manner.
    This is just for testing; it is not part of the QKV spec.
    """
    row = fp16(rng.random(seqLen))
    denom = np.float16(0.0)
    for k in range(seqLen):
        denom = np.float16(denom + np.float16(row[k]))
    return fp16(row / denom)


def main(N_ROWS = 8) -> None:
    np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

    rng = np.random.default_rng(0)
    qkv = QKV()

    Dh = qkv.dh
    seqLen = qkv.seqLen
    slices = seqLen // 32

    errs = []
    for r in range(N_ROWS):
        soft = make_softmax_row_fp16(rng, seqLen)
        v = fp16(rng.normal(0.0, 1.0, size=(seqLen, Dh)))

        out = run_one_row(qkv, soft, v)
        gold = golden_row_v_softmax_tree_blocked(soft, v)

        e = mixed_err(out, gold)
        errs.append(e)

        print(f"[Row {r}] mixed_err={e:.6e}")

    max_e = max(errs) if errs else 0.0
    print("=== PATTERN CHECK: QKV N rows ===")
    print(f"N_ROWS={N_ROWS}, max_mixed_err={max_e:.6e}")

    # Threshold: tune based on how strictly you want to match the exact tree.
    assert max_e < 5e-2, "At least one row mismatched too much."

    expected_calls = N_ROWS * slices * (1 + Dh)
    print(f"trgCnt={qkv.trgCnt}, expected_calls={expected_calls}")
    assert qkv.trgCnt == expected_calls, "trgCnt mismatch."


if __name__ == "__main__":
    main()
