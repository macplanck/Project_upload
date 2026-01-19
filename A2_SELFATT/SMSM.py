from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np 


# ============================================================
# Exceptions (explicit, actionable)
# ============================================================
class ExponentialError(FloatingPointError):
    """Raised when exp() produces inf/NaN after FP16 quantization."""


class UnderflowErrorFP16(FloatingPointError):
    """
    Raised when a nonzero value becomes zero after checked FP16 scaling:
        term = fp16(x / 2**k)
        if x != 0 and term == 0 -> underflow
    """


class OverflowErrorFP16(FloatingPointError):
    """
    Raised when a checked FP16 scaling becomes non-finite after FP16 quantization:
        term = fp16(x / 2**k)
        if not isfinite(term) -> overflow/invalid
    """


class ShiftError(ValueError):
    """Raised when shift violates spec constraints."""


class NewMaxError(ValueError):
    """Raised when newMax leaves the unbiased FP16 normal range [-14, 15]."""


class NewRsltError(FloatingPointError):
    """Raised when newRslt becomes non-finite."""


# ============================================================
# FP16 bit utilities
# ============================================================
def fp16_bits(x: np.float16) -> int:
    """Return raw uint16 bit pattern for FP16 value."""
    return int(np.frombuffer(np.float16(x).tobytes(), dtype=np.uint16)[0])


def is_fp16_subnormal(x: np.float16) -> bool:
    """
    FP16 subnormal iff exponent field == 0 and fraction != 0.
    """
    bits = fp16_bits(x)
    exp_field = (bits >> 10) & 0x1F
    frac_field = bits & 0x03FF
    return (exp_field == 0) and (frac_field != 0)


def fp16_biased_exp_field(x: np.float16) -> int:
    """Extract the 5-bit biased exponent field [0..31]."""
    return (fp16_bits(x) >> 10) & 0x1F


# ============================================================
# Checked scaling primitive (your specified rule)
# ============================================================
def checked_scale_pow2_fp16(x_fp16: np.float16, k: int) -> np.float16:
    """
    Compute term = fp16(x / 2**k) with strict checks:

    Rule:
      - term = fp16(x / 2**k)
      - if x != 0 and term == 0 -> raise UnderflowErrorFP16
      - if not isfinite(term)   -> raise OverflowErrorFP16

    Notes:
      - k is an integer shift amount (allowed non-FP16 operation).
      - division itself is performed in higher precision then quantized to FP16.
    """
    term = np.float16(np.float32(x_fp16) / (2.0 ** int(k)))

    if (x_fp16 != np.float16(0.0)) and (term == np.float16(0.0)):
        # raise UnderflowErrorFP16(f"Underflow: x={x_fp16} / 2**{k} -> term=0 (FP16).")
        term = np.float16(0.0)

    if not np.isfinite(term):
        raise OverflowErrorFP16(f"Overflow/Invalid: x={x_fp16} / 2**{k} -> term={term} (FP16).")

    return term


# ============================================================
# Core class
# ============================================================
@dataclass(frozen=True)
class SMSMOutput:
    """
    Output of one_operation().
    """
    maxExp:   int
    sumDnm:   np.float16
    exped: np.float16
    DmnValid: bool
    trgCnt:   int

@dataclass
class SMSM:
    """
    SMSM: Streaming Max + Scaled-denominator accumulator.

    Maintains:
      maxExp (int): running max unbiased exponent of exped = fp16(exp(kqed))
      sumDnm (fp16): sum(exped) / 2**maxExp

    Control:
      colCnt: [0..seqLen-1]
      trgCnt: increments each one_operation()

    Conventions:
      - All floating ops are quantized to FP16.
      - Shift amounts are integer (2**k uses integer k).
      - exped == 0 (exp underflow or subnormal flush) is acceptable and should not raise.
    """
    seqLen: int = 8192

    colCnt: int = 0
    trgCnt: int = 0

    maxExp: int = -14
    sumDnm: np.float16 = np.float16(0.0)

    def one_operation(self, kqed: np.float16) -> Tuple[int, np.float16, np.float16]:
        """
        One cycle update.

        Returns:
          (newMax, newRslt, exped)
        """
        newMax, newRslt, exped = self.eval(kqed)
        self.maxExp = int(newMax)
        self.sumDnm = np.float16(newRslt)
        DmnValid = self.update_ctrl()
        self.trgCnt += 1
        return SMSMOutput(maxExp = int(newMax), sumDnm = np.float16(newRslt), exped = np.float16(exped), DmnValid = DmnValid, trgCnt = self.trgCnt)

    def eval(self, kqed: np.float16) -> Tuple[int, np.float16, np.float16]:
        """
        Implements the provided pseudocode with:
          - FP16 bit-accurate subnormal detection (flush to 0)
          - exponent extraction from FP16 bits
          - checked scaling for x / 2**k

        Returns:
          (newMax, newRslt, exped)
        """
        kqed_fp16 = np.float16(kqed)

        # exp computed in higher precision, then quantized to FP16
        exped = np.float16(np.exp(np.float32(kqed_fp16)))
        # print(exped)

        # flush FP16 subnormals to 0
        if is_fp16_subnormal(exped):
            exped = np.float16(0.0)

        # inf/nan -> error
        if not np.isfinite(exped):
            raise ExponentialError(f"exped is non-finite after FP16 quantization: exped={exped}")

        # early exit: exped == 0 is acceptable
        if exped == np.float16(0.0):
            newMax = int(self.maxExp)
            newRslt = np.float16(self.sumDnm)
            # Still validate result finiteness
            if not np.isfinite(newRslt):
                raise NewRsltError(f"newRslt non-finite on exped==0 early path: {newRslt}")
            return newMax, newRslt, exped

        # exponent extraction (biased) -> unbiased candidate
        biased = fp16_biased_exp_field(exped)
        # biased==0 would imply zero/subnormal (handled); biased==31 implies inf/nan (handled)
        candidate = int(biased) - 15

        if self.colCnt == 0:
            newMax = candidate
            newRslt = checked_scale_pow2_fp16(exped, newMax)

        elif self.maxExp < candidate:
            shift = int(candidate - self.maxExp)
            if (shift > 29) or (shift < 1):
                raise ShiftError(f"Invalid shift={shift} (candidate={candidate}, maxExp={self.maxExp})")

            termA = checked_scale_pow2_fp16(self.sumDnm, shift)
            termB = checked_scale_pow2_fp16(exped, candidate)

            newMax = candidate
            newRslt = np.float16(np.float16(termA) + np.float16(termB))

        else:
            newMax = int(self.maxExp)
            termB = checked_scale_pow2_fp16(exped, newMax)
            newRslt = np.float16(np.float16(self.sumDnm) + np.float16(termB))

        # newMax guard
        if (newMax > 15) or (newMax < -14):
            raise NewMaxError(f"newMax out of range [-14, 15]: newMax={newMax}")

        # newRslt guard
        if not np.isfinite(newRslt):
            raise NewRsltError(f"newRslt became non-finite: newRslt={newRslt}")

        return int(newMax), np.float16(newRslt), np.float16(exped)

    def update_ctrl(self) -> None:
        """Update colCnt and reset row state at end-of-row."""
        if self.colCnt == (self.seqLen - 1):
            self.colCnt = 0
            self.maxExp = -14
            self.sumDnm = np.float16(0.0)
            return 1
        else:
            self.colCnt += 1
            return 0


# ============================================================
# main: pattern checks on return values
# ============================================================
def verify_smsm_step(
    *,
    prevMax: int,
    prevSum: np.float16,
    colUsed: int,
    newMax: int,
    newRslt: np.float16,
    exped: np.float16,
    colAfter: int,
    seqLen: int,
) -> None:
    """
    Self-verifying pattern checker for one SMSM step.

    Raises AssertionError on ANY spec violation.
    """

    # ---- type checks ----
    assert isinstance(newMax, int), "newMax must be int"
    assert isinstance(newRslt, np.float16), "newRslt must be fp16"
    assert isinstance(exped, np.float16), "exped must be fp16"

    # ---- exponent range ----
    assert -14 <= newMax <= 15, f"newMax out of range: {newMax}"

    # ---- finiteness ----
    assert np.isfinite(newRslt), f"newRslt not finite: {newRslt}"
    assert np.isfinite(exped) or exped == np.float16(0), f"exped invalid: {exped}"

    # ---- exped == 0 rule ----
    if exped == np.float16(0) and colUsed != seqLen - 1:
        assert newMax == prevMax, "exped==0 must not update maxExp"
        assert newRslt == prevSum, "exped==0 must not update sumDnm"

    # ---- column progression ----
    if colUsed == seqLen - 1:
        assert colAfter == 0, "colCnt must reset at end of row"
    else:
        assert colAfter == colUsed + 1, "colCnt must increment by 1"

def main() -> None:
    np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

    smsm = SMSM(seqLen=8192)

    inputs = [
        np.float16(-30.0),  # exp underflow → 0 (legal)
        np.float16(-12.0),
        np.float16(-4.0),
        np.float16(0.0),
        np.float16(2.0),
        np.float16(6.0),
    ]

    print("idx col  kqed     exped     newMax  newRslt  colAfter")

    for i in range(8192*3):
        x = inputs[i % len(inputs)]

        colUsed = smsm.colCnt
        prevMax = smsm.maxExp
        prevSum = smsm.sumDnm

        newMax, newRslt, exped = smsm.one_operation(x)

        # self-verification pattern
        verify_smsm_step(
            prevMax=prevMax,
            prevSum=prevSum,
            colUsed=colUsed,
            newMax=newMax,
            newRslt=newRslt,
            exped=exped,
            colAfter=smsm.colCnt,
            seqLen=smsm.seqLen,
        )

        print(
            f"{i:>3} {colUsed:>3} {float(x):>7.2f} "
            f"{float(exped):>8.4f} {newMax:>6} "
            f"{float(newRslt):>8.5f} {smsm.colCnt:>8}"
        )

    print("\nALL PATTERNS PASSED")

if __name__ == "__main__":
    main()