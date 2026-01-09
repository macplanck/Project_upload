from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


class SmdDenominatorError(RuntimeError):
    """Raised when smDmn is invalid (0/subnormal/NaN/Inf) in FP16."""


class Fp16NumericError(RuntimeError):
    """Raised when any intermediate or output becomes NaN/Inf in FP16."""


def fp16(x) -> np.float16:
    """
    Cast to IEEE-754 binary16 (FP16).

    Notes
    -----
    This function returns a real FP16 type (numpy.float16).
    All arithmetic in this module is intended to be performed on np.float16
    so that operations round to FP16 at each step.
    """
    return np.float16(x)


def _is_fp16_nan_or_inf(x: np.float16) -> bool:
    """True if FP16 value is NaN or +/-Inf."""
    # np.isfinite works for np.float16
    return not bool(np.isfinite(x))


def _is_fp16_zero(x: np.float16) -> bool:
    """True if FP16 value is +0 or -0."""
    # Comparing to 0.0 is safe for float16 here; preserves -0 == 0 behavior.
    return bool(x == fp16(0.0))


def _is_fp16_subnormal(x: np.float16) -> bool:
    """
    True if FP16 is subnormal (exp==0 and frac!=0).

    Implementation detail:
    - binary16 layout: sign(1) exp(5) frac(10)
    - exp==0 and frac!=0 => subnormal
    """
    bits = np.frombuffer(np.float16(x).tobytes(), dtype=np.uint16)[0]
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x03FF
    return bool((exp == 0) and (frac != 0))


def _is_fp16_invalid_denominator(x: np.float16) -> bool:
    """
    Denominator invalid if:
    - 0 (+0 or -0)
    - subnormal
    - NaN or Inf
    """
    if _is_fp16_nan_or_inf(x):
        return True
    if _is_fp16_zero(x):
        return True
    if _is_fp16_subnormal(x):
        return True
    return False


@dataclass
class SMD:
    """
    SMD — Softmax Division block simulator.

    Purpose
    -------
    Perform:
        newDiv = (fp16(inNmn) / fp16(2.0 ** smExp)) / smDmn
    with FP16 arithmetic and error checks.

    Model parameters
    ----------------
    seqLen : int
        Row length. Default 8192.

    State / control
    ---------------
    colCnt : int
        Current column index within the row [0, seqLen-1].
    trgCnt : int
        Increments once per one_operation() call.

    Storage (latched at colCnt==0)
    ------------------------------
    smExp : int
        Unbiased exponent for power-of-two scaling.
    smDmn : np.float16
        Softmax denominator in FP16.
    """

    seqLen: int = 8192
    colCnt: int = 0
    trgCnt: int = 0

    smExp: int = 0
    smDmn: np.float16 = fp16(1.0)

    # Documentation-only latency parameter (not cycle-accurate in this model).
    lat: int = 2

    def one_operation(self, inNmn: np.float16, *, inExp: Optional[int] = None, inDmn: Optional[np.float16] = None) -> np.float16:
        """
        Execute one operation.

        Contract
        --------
        - When colCnt == 0, caller must provide valid inExp and inDmn.

        Parameters
        ----------
        inNmn : np.float16
            Numerator input (FP16).
        inExp : Optional[int]
            New smExp (int), required when colCnt == 0.
        inDmn : Optional[np.float16]
            New smDmn (FP16), required when colCnt == 0.

        Returns
        -------
        np.float16
            newDiv in FP16.

        Raises
        ------
        ValueError
            If inExp/inDmn missing when colCnt==0.
        SmdDenominatorError
            If smDmn is invalid.
        Fp16NumericError
            If any intermediate/output is NaN/Inf.
        """
        if self.colCnt == 0:
            if inExp is None or inDmn is None:
                raise ValueError("one_operation requires inExp and inDmn when colCnt == 0.")
            self.smExp = int(inExp)
            self.smDmn = fp16(inDmn)

        newDiv = self.eval(fp16(inNmn))
        self.update_ctrl()
        self.trgCnt += 1
        return newDiv

    def eval(self, inNmn: np.float16) -> np.float16:
        """
        Compute:
            pow2 = fp16(2.0 ** smExp)
            q1   = fp16(inNmn) / pow2
            newDiv = q1 / smDmn
        Raise errors on:
            - invalid denominator: 0/subnormal/NaN/Inf
            - NaN/Inf in pow2, q1, or newDiv
        """
        if _is_fp16_invalid_denominator(self.smDmn):
            raise SmdDenominatorError(f"SMD denominator error: smDmn={self.smDmn!r} (FP16 invalid).")

        pow2 = fp16(fp16(2.0) ** int(self.smExp))  # ensure exponent is int, result FP16

        q1 = fp16(inNmn) / pow2
        q1 = fp16(q1)  # force FP16 rounding explicitly

        newDiv = q1 / self.smDmn
        newDiv = fp16(newDiv)  # force FP16 rounding explicitly

        if _is_fp16_nan_or_inf(pow2) or _is_fp16_nan_or_inf(q1) or _is_fp16_nan_or_inf(newDiv):
            raise Fp16NumericError(
                f"FP16 numeric error: pow2={pow2!r}, q1={q1!r}, newDiv={newDiv!r}"
            )

        return newDiv

    def update_ctrl(self) -> None:
        """Update colCnt (wrap at seqLen-1)."""
        if self.colCnt == (self.seqLen - 1):
            self.colCnt = 0
        else:
            self.colCnt += 1


def main(seed=0) -> None:
    """
    Pattern check for SMD.

    Strategy
    --------
    1) Generate a row of FP16 numerators.
    2) Choose a valid smExp and a valid (non-subnormal, nonzero) denominator smDmn.
    3) Run SMD across one full row and compare against a golden FP16 computation
       that quantizes at each step in the same places.

    Also includes negative tests:
    - smDmn == 0 -> must raise SmdDenominatorError
    - smDmn subnormal -> must raise SmdDenominatorError
    """
    rng = np.random.default_rng(seed)

    seqLen = 128  # use a smaller row for quick pattern check
    smd = SMD(seqLen=seqLen)

    # Create FP16 row numerators
    inNmns = fp16(rng.normal(loc=0.0, scale=1.0, size=seqLen).astype(np.float16))

    # Choose exponent and denominator
    inExp = 3  # scale by 2^3 = 8
    inDmn = fp16(3.5)  # must be normal and nonzero

    # Golden function with explicit FP16 quantization points
    def golden(inNmn_fp16: np.float16, exp_int: int, dmn_fp16: np.float16) -> np.float16:
        pow2 = fp16(fp16(2.0) ** int(exp_int))
        q1 = fp16(fp16(inNmn_fp16) / pow2)
        out = fp16(q1 / fp16(dmn_fp16))
        # same NaN/Inf rule
        if _is_fp16_nan_or_inf(pow2) or _is_fp16_nan_or_inf(q1) or _is_fp16_nan_or_inf(out):
            raise Fp16NumericError("Golden FP16 numeric error.")
        return out

    # Run and compare one full row
    hw_out = []
    gd_out = []

    for i in range(seqLen):
        if smd.colCnt == 0:
            y = smd.one_operation(inNmns[i], inExp=inExp, inDmn=inDmn)
        else:
            y = smd.one_operation(inNmns[i])
        hw_out.append(y)

        gd_out.append(golden(inNmns[i], inExp, inDmn))

    hw_out = np.asarray(hw_out, dtype=np.float16)
    gd_out = np.asarray(gd_out, dtype=np.float16)

    mism = np.where(hw_out.view(np.uint16) != gd_out.view(np.uint16))[0]
    print("=== PATTERN CHECK: NORMAL CASE ===")
    print(f"seqLen={seqLen}, inExp={inExp}, inDmn={inDmn}")
    print(f"mismatch_count={mism.size}")

    if mism.size > 0:
        k = int(mism[0])
        print(f"first mismatch idx={k}")
        print(f"  inNmn={inNmns[k]!r}")
        print(f"  hw={hw_out[k]!r}, gd={gd_out[k]!r}")
        print(f"  hw_bits=0x{hw_out.view(np.uint16)[k]:04x}, gd_bits=0x{gd_out.view(np.uint16)[k]:04x}")
        raise AssertionError("PATTERN CHECK FAILED: outputs differ at FP16 bit level.")
    else:
        print("PASS: bit-accurate match vs golden.")

    # Negative test: denominator == 0
    smd2 = SMD(seqLen=8)
    try:
        _ = smd2.one_operation(fp16(1.0), inExp=0, inDmn=fp16(0.0))
        raise AssertionError("Expected SmdDenominatorError for smDmn==0, but no error was raised.")
    except SmdDenominatorError:
        print("PASS: smDmn==0 correctly raises SmdDenominatorError.")

    # Negative test: denominator subnormal (smallest positive subnormal is 2^-24 for FP16)
    # Construct a subnormal via bits: exp=0, frac=1
    sub_bits = np.uint16(0x0001)
    sub = np.frombuffer(sub_bits.tobytes(), dtype=np.float16)[0]
    smd3 = SMD(seqLen=8)
    try:
        _ = smd3.one_operation(fp16(1.0), inExp=0, inDmn=sub)
        raise AssertionError("Expected SmdDenominatorError for smDmn subnormal, but no error was raised.")
    except SmdDenominatorError:
        print("PASS: smDmn subnormal correctly raises SmdDenominatorError.")


if __name__ == "__main__":
    for i in range(100):
        main(i)
