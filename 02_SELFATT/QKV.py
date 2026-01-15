from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================
# Errors (explicit, spec-oriented)
# ============================================================

class SpecViolationError(RuntimeError):
    """Raised when the internal schedule/control sequence violates the spec."""


class FPVOverflowError(OverflowError):
    """Raised when FP16 exponent adjustment would produce an invalid/NaN exponent field."""


# ============================================================
# FP16 bit packing/unpacking (IEEE-754 binary16)
# ============================================================

@dataclass(frozen=True)
class FP16Fields:
    """
    IEEE-754 binary16 fields.

    Attributes
    ----------
    signed : int
        0 or 1.
    exponent : int
        5-bit biased exponent field in [0..31].
        0 => zero/subnormal, 31 => inf/NaN.
    mantissa : int
        10-bit fraction field in [0..1023].
    """
    signed: int
    exponent: int
    mantissa: int


def fp16_unpack(x: np.float16) -> FP16Fields:
    """
    Unpack np.float16 into IEEE-754 binary16 fields.

    Parameters
    ----------
    x : np.float16
        Input half-precision float.

    Returns
    -------
    FP16Fields
        The sign, biased exponent, and mantissa fields.

    Example
    -------
    >>> f = fp16_unpack(np.float16(1.5))
    >>> (f.signed, f.exponent, f.mantissa)  # doctest: +SKIP
    (0, 15, 512)
    """
    u = np.frombuffer(np.float16(x).tobytes(), dtype=np.uint16)[0]
    signed = (u >> 15) & 0x1
    exponent = (u >> 10) & 0x1F
    mantissa = u & 0x03FF
    return FP16Fields(int(signed), int(exponent), int(mantissa))


def fp16_pack(signed: int, exponent: int, mantissa: int) -> np.float16:
    """
    Pack IEEE-754 binary16 fields into np.float16.

    Parameters
    ----------
    signed : int
        0 or 1.
    exponent : int
        5-bit biased exponent field in [0..31].
    mantissa : int
        10-bit fraction field in [0..1023].

    Returns
    -------
    np.float16
        Packed half-precision float.

    Notes
    -----
    This function does not attempt to canonicalize NaNs; it just packs bits.
    """
    if signed not in (0, 1):
        raise ValueError("signed must be 0 or 1")
    if not (0 <= exponent <= 31):
        raise ValueError("exponent must be in [0..31]")
    if not (0 <= mantissa <= 1023):
        raise ValueError("mantissa must be in [0..1023]")

    u = (np.uint16(signed) << 15) | (np.uint16(exponent) << 10) | np.uint16(mantissa)
    return np.frombuffer(np.uint16(u).tobytes(), dtype=np.float16)[0]


def fp16(x: Union[float, np.float16, np.ndarray]) -> Union[np.float16, np.ndarray]:
    """
    Cast helper to enforce FP16 operations.

    Parameters
    ----------
    x : float or np.float16 or np.ndarray
        Value(s) to cast.

    Returns
    -------
    np.float16 or np.ndarray
        FP16-cast value(s).
    """
    return np.asarray(x, dtype=np.float16) if isinstance(x, np.ndarray) else np.float16(x)


def adder_tree_fp16(vec32: np.ndarray) -> np.float16:
    """
    FP16 pairwise reduction (tree) over a vector.

    Parameters
    ----------
    vec32 : np.ndarray
        Shape (32,), dtype float16.

    Returns
    -------
    np.float16
        Reduced sum in FP16, using pairwise (tree) adds.

    Notes
    -----
    This intentionally mimics a tree reduction rather than a linear fold.
    """
    if vec32.shape != (32,):
        raise ValueError("adder_tree_fp16 expects shape (32,)")
    if vec32.dtype != np.float16:
        vec32 = np.asarray(vec32, dtype=np.float16)

    lvl = vec32
    while lvl.size > 1:
        nxt: List[np.float16] = []
        i = 0
        while i + 1 < lvl.size:
            nxt.append(np.float16(lvl[i] + lvl[i + 1]))
            i += 2
        if i < lvl.size:
            nxt.append(np.float16(lvl[i]))
        lvl = np.asarray(nxt, dtype=np.float16)
    return np.float16(lvl[0])


# ============================================================
# QKV simulator
# ============================================================

class QKV:
    """
    QKV block simulator (V * softmax(QK^T) row generator) using a softmax(QK^T)-stationary schedule.

    What this implements, restated
    ------------------------------
    - For each slice of the sequence dimension (32-wide, total seqLen/32 slices):
      1) Load QKbuf (32 fp16 values): one operation where QKFlag==1.
      2) Then perform Dh operations (Dh = hiddenSize/headNum):
         each operation streams V[kBlock:kBlock+32, d] as 32 int8 values,
         dequantizes by exponent adjustment using dqFac, computes dot32(QKbuf, Vdeq),
         adds previous accu[d], and writes back accu[d].
    - Outputs are considered valid only on the *final slice* and only for V-ops (QKFlag==0),
      one column per cycle.

    Model parameters
    ---------------
    hiddenSize : int
        Default 4096.
    headNum : int
        Default 32.
    seqLen : int
        Default 8192.
    Dh : int
        Derived = hiddenSize/headNum.

    Control state
    -------------
    sliceCnt : int
        Which seq slice we are on, range [0..(seqLen/32 - 1)].
    colCnt : int
        Which output column d we are on, range [0..(Dh - 1)].
    QKFlag : int
        1 => input is QKbuf (softmax slice), 0 => input is V block.
        Initialized high per spec.

    Outputs
    -------
    outValid : bool
        True exactly when (not QKFlag) and sliceCnt is the last slice.
    outCol : int
        Column index associated with outValid.

    Notes on numeric behavior (per spec intent)
    -------------------------------------------
    - All arithmetic is performed in float16.
    - V dequantization is implemented by FP16 *exponent field adjustment* (no pow/div).
    - If exponent adjustment would underflow into subnormal/zero: FTZ => 0.
    - If exponent adjustment would exceed max normal exponent field (30): raise FPVOverflowError.
    """

    def __init__(self) -> None:
        # Reminder settings: treat overflow/div/invalid as exceptions, underflow ignored (per your reminder).
        np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

        # Model parameters
        self.hiddenSize: int = 4096
        self.headNum: int = 32
        self.seqLen: int = 8192
        self.lat: int = 4

        if self.hiddenSize % self.headNum != 0:
            raise ValueError("hiddenSize must be divisible by headNum")
        self.Dh: int = self.hiddenSize // self.headNum

        if self.seqLen % 32 != 0:
            raise ValueError("seqLen must be divisible by 32")
        self.slices: int = self.seqLen // 32

        # Control state
        self.sliceCnt: int = 0
        self.colCnt: int = 0
        self.QKFlag: int = 1  # initialized high

        # Cycle counter
        self.trgCnt: int = 0

        # Storage
        self.QKbuf: np.ndarray = np.zeros((32,), dtype=np.float16)
        self.Vdeq: np.ndarray = np.zeros((32,), dtype=np.float16)
        self.accu: np.ndarray = np.zeros((self.Dh,), dtype=np.float16)

        # Input latches (to be set before one_operation)
        self.inNum: Optional[Union[np.ndarray, Sequence[Union[np.float16, float, int]]]] = None
        self.dqFac: Optional[int] = None  # signed int, 5-bit in HW

        # Output signals
        self.outValid: bool = False
        self.outCol: int = 0

        # Internal schedule checker (enforce: per slice, 1x QK then Dh x V)
        self._phase: str = "QK"  # "QK" or "V"
        self._expected_v_col: int = 0
        self._expected_slice: int = 0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def set_inputs(self, *, inNum: Sequence[Union[np.float16, float, int]], dqFac: int) -> None:
        """
        Set the inputs for the next one_operation() call.

        Parameters
        ----------
        inNum : sequence
            Length-32 vector.
            - If QKFlag==1 for the coming cycle: elements are interpreted as fp16 softmax values.
            - If QKFlag==0 for the coming cycle: elements are interpreted as int8 V values.
        dqFac : int
            Signed shift count applied to FP16 biased exponent during V dequantization.

        Notes
        -----
        The caller is responsible for providing the correct data type per phase, consistent with QKFlag.
        """
        arr = np.asarray(inNum)
        if arr.shape != (32,):
            raise ValueError("inNum must have shape (32,)")
        self.inNum = arr
        self.dqFac = int(dqFac)

    def one_operation(self) -> np.float16:
        """
        Perform one cycle operation.

        Returns
        -------
        np.float16
            prtSum (partialSum in your spec). For QK cycles this is 0 (harmless).
            For V cycles this is the updated accumulator value for the current colCnt.

        Side effects
        ------------
        - Updates QKbuf or accu[colCnt] depending on QKFlag.
        - Updates outValid/outCol and control state (sliceCnt/colCnt/QKFlag).
        - Increments trgCnt by 1.
        """
        prtSum = self.eval()

        if self.QKFlag == 0:
            self.accu[self.colCnt] = prtSum

        self.update_ctrl()
        self.trgCnt += 1

        return prtSum

    # ------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------

    def eval(self) -> np.float16:
        """
        Evaluate the datapath for this cycle.

        Returns
        -------
        np.float16
            partialSum per the spec.

        Raises
        ------
        SpecViolationError
            If the per-slice schedule order is violated.
        FPVOverflowError
            If exponent adjustment would exceed FP16 max normal exponent field.
        """
        if self.inNum is None or self.dqFac is None:
            raise SpecViolationError("Inputs not set before eval()")

        # Enforce schedule: per slice, 1 QK op then Dh V ops.
        self._assert_schedule()

        if self.QKFlag == 1:
            # Load QKbuf
            self.QKbuf = np.asarray(self.inNum, dtype=np.float16)
            partialSum = np.float16(0.0)  # harmless, to keep return type consistent
            return partialSum

        # V path: dequantize by FP16 exponent adjustment (FTZ on subnormals)
        dqFac = int(self.dqFac)

        # Interpret inNum as int8 for V input (spec says 32*int8 expected)
        v_int8 = np.asarray(self.inNum, dtype=np.int8)

        for i in range(32):
            v_fp16 = np.float16(v_int8[i])  # int8 -> fp16 (exact for [-128..127])
            f = fp16_unpack(v_fp16)

            # If original is zero/subnormal, FTZ => 0
            if f.exponent == 0:
                self.Vdeq[i] = np.float16(0.0)
                continue

            fpVExpo = int(f.exponent) - dqFac  # biased exponent adjustment

            # FTZ behavior: if exponent field would be <= 0, flush to zero.
            if fpVExpo <= 0:
                self.Vdeq[i] = np.float16(0.0)
            elif fpVExpo > 30:
                # Spec comment: "NaN occur" -> raise
                raise FPVOverflowError(
                    f"Exponent adjust overflow: exp={f.exponent}, dqFac={dqFac}, fpVExpo={fpVExpo}"
                )
            else:
                self.Vdeq[i] = fp16_pack(f.signed, fpVExpo, f.mantissa)

        # Dot + accumulator (all FP16)
        prod = np.asarray(self.Vdeq * self.QKbuf, dtype=np.float16)
        dot = adder_tree_fp16(prod)
        partialSum = np.float16(dot + np.float16(self.accu[self.colCnt]))
        return partialSum

    def update_ctrl(self) -> None:
        """
        Update control FSM and output flags per spec pseudocode.
        """
        # Output flags are derived from the *current* state (before transitions)
        self.outValid = (self.QKFlag == 0) and (self.sliceCnt == (self.slices - 1))
        self.outCol = int(self.colCnt)

        if self.QKFlag == 1:
            self.QKFlag = 0
            # colCnt stays (should be 0 at the start of V phase)
        elif self.colCnt == (self.Dh - 1):
            self.QKFlag = 1
            self.colCnt = 0

            if self.sliceCnt == (self.slices - 1):
                self.sliceCnt = 0
                self.accu[:] = np.float16(0.0)
            else:
                self.sliceCnt += 1
        else:
            self.colCnt += 1

    # ------------------------------------------------------------
    # Internal schedule checker
    # ------------------------------------------------------------

    def _assert_schedule(self) -> None:
        """
        Internal assertion checker enforcing:
          per slice: 1x QK call then Dh x V calls.

        This checks:
        - sliceCnt aligns with expected slice index
        - QKFlag matches expected phase
        - during V phase, colCnt marches 0..Dh-1
        """
        if self.sliceCnt != self._expected_slice:
            raise SpecViolationError(
                f"sliceCnt mismatch: got {self.sliceCnt}, expected {self._expected_slice}"
            )

        if self._phase == "QK":
            if self.QKFlag != 1:
                raise SpecViolationError("Expected QK phase but QKFlag!=1")
            if self.colCnt != 0:
                # Per spec, after finishing a slice, colCnt resets to 0 before next QK.
                raise SpecViolationError(f"Expected colCnt==0 at QK phase, got {self.colCnt}")
            # After this eval, phase should move to V (Dh entries)
            self._phase = "V"
            self._expected_v_col = 0
            return

        # V phase
        if self.QKFlag != 0:
            raise SpecViolationError("Expected V phase but QKFlag!=0")
        if self.colCnt != self._expected_v_col:
            raise SpecViolationError(
                f"V phase colCnt mismatch: got {self.colCnt}, expected {self._expected_v_col}"
            )

        # After consuming this V op, advance expected col
        self._expected_v_col += 1
        if self._expected_v_col >= self.Dh:
            # Next op must be QK of the next slice
            self._phase = "QK"
            if self._expected_slice == (self.slices - 1):
                self._expected_slice = 0
            else:
                self._expected_slice += 1


# ============================================================
# Reference computation for checking correctness
# ============================================================

def deq_int8_to_fp16_with_expadj(v_int8: np.ndarray, dqFac: int) -> np.ndarray:
    """
    Reference: apply the same FP16 exponent-adjust dequantization used in QKV.eval().

    Parameters
    ----------
    v_int8 : np.ndarray
        Shape (N,), dtype int8.
    dqFac : int
        Signed exponent shift count.

    Returns
    -------
    np.ndarray
        Shape (N,), dtype float16.
    """
    out = np.zeros_like(v_int8, dtype=np.float16)
    for i in range(v_int8.size):
        v_fp16 = np.float16(np.int8(v_int8[i]))
        f = fp16_unpack(v_fp16)

        if f.exponent == 0:
            out[i] = np.float16(0.0)
            continue

        fpVExpo = int(f.exponent) - int(dqFac)
        if fpVExpo <= 0:
            out[i] = np.float16(0.0)
        elif fpVExpo > 30:
            raise FPVOverflowError(f"Overflow in reference deq: exp={f.exponent}, dqFac={dqFac}")
        else:
            out[i] = fp16_pack(f.signed, fpVExpo, f.mantissa)
    return out


def main() -> None:
    """
    Minimal pattern to check:
    - control sequence correctness (asserted internally)
    - numeric equivalence (FP16) between streamed accumulation and a reference computation
    - outValid/outCol behavior on the last slice
    """
    np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

    rng = np.random.default_rng(0)
    qkv = QKV()

    Dh = qkv.Dh
    seqLen = qkv.seqLen
    slices = qkv.slices

    # Build one "softmax row" (non-negative, normalized) in FP16.
    softmax_row = fp16(rng.random(seqLen))
    denom = np.float16(0.0)
    for k in range(seqLen):
        denom = np.float16(denom + softmax_row[k])
    softmax_row = fp16(softmax_row / denom)

    # Build V as int8: (seqLen, Dh)
    # Keep values modest to reduce overflow risk in exp-adjust for random dqFac.
    v_mat = rng.integers(low=-32, high=32, size=(seqLen, Dh), dtype=np.int16).astype(np.int8)

    # Pick a dqFac (signed 5-bit). This is a stress knob: positive => more FTZ.
    dqFac = int(rng.integers(low=-3, high=6))

    # Drive the QKV schedule exactly: for each slice: 1 QK load then Dh V ops
    outputs_streamed: List[np.float16] = []
    outcols_streamed: List[int] = []

    for s in range(slices):
        # QK load op
        qk_slice = softmax_row[s * 32 : (s + 1) * 32]
        qkv.set_inputs(inNum=qk_slice, dqFac=dqFac)
        _ = qkv.one_operation()

        # Dh V ops for this slice
        for d in range(Dh):
            v_block = v_mat[s * 32 : (s + 1) * 32, d]
            qkv.set_inputs(inNum=v_block, dqFac=dqFac)
            prt = qkv.one_operation()

            # Collect final outputs only when outValid asserts (last slice only)
            if qkv.outValid:
                outputs_streamed.append(prt)
                outcols_streamed.append(qkv.outCol)

    # We should have exactly Dh outputs, with outCol walking 0..Dh-1
    if len(outputs_streamed) != Dh:
        raise AssertionError(f"Expected {Dh} valid outputs, got {len(outputs_streamed)}")

    if outcols_streamed != list(range(Dh)):
        raise AssertionError(f"outCol sequence mismatch: got {outcols_streamed[:10]} ...")

    # Reference computation:
    # For each d: sum_k softmax_row[k] * deq(v_mat[k,d], dqFac) using the same exp-adjust logic.
    ref = np.zeros((Dh,), dtype=np.float16)
    for d in range(Dh):
        v_deq = deq_int8_to_fp16_with_expadj(v_mat[:, d], dqFac)
        # FP16 dot via slice-wise tree reductions to mimic the streaming behavior more closely.
        acc = np.float16(0.0)
        for s in range(slices):
            a = softmax_row[s * 32 : (s + 1) * 32]
            b = v_deq[s * 32 : (s + 1) * 32]
            dot = adder_tree_fp16(fp16(a * b))
            acc = np.float16(acc + dot)
        ref[d] = acc

    got = np.asarray(outputs_streamed, dtype=np.float16)

    # Exact FP16 equality should hold because we used identical operations/order on both sides.
    if not np.array_equal(got, ref):
        # Provide a helpful diff summary
        mismatch = np.nonzero(got != ref)[0]
        i0 = int(mismatch[0])
        raise AssertionError(
            f"Mismatch at col {i0}: got={got[i0]} ref={ref[i0]} "
            f"(total mismatches={mismatch.size})"
        )

    print("PASS: schedule + numeric checks OK.")
    print(f"dqFac={dqFac}, trgCnt={qkv.trgCnt}, lat={qkv.lat}")


if __name__ == "__main__":
    main()
