from __future__ import annotations

from dataclasses import dataclass
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Tuple
import math
import numpy as np 
=======
from typing import Tuple, List
import numpy as np
>>>>>>> orgin/main
=======
from typing import Tuple
import numpy as np 
>>>>>>> orgin/main


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


<<<<<<< HEAD
def fp16_unbiased_exp_if_normal(x: np.float16) -> int | None:
    """
    Return unbiased exponent if x is a *normal* FP16 number.
    Return None if x is zero or subnormal (exp_field == 0), or inf/nan (exp_field == 31).
    """
    biased = fp16_biased_exp_field(x)
    if biased == 0 or biased == 31:
        return None
    return int(biased) - 15


=======
>>>>>>> orgin/main
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
<<<<<<< HEAD
        raise UnderflowErrorFP16(f"Underflow: x={x_fp16} / 2**{k} -> term=0 (FP16).")
=======
        # raise UnderflowErrorFP16(f"Underflow: x={x_fp16} / 2**{k} -> term=0 (FP16).")
        term = np.float16(0.0)
>>>>>>> orgin/main

    if not np.isfinite(term):
        raise OverflowErrorFP16(f"Overflow/Invalid: x={x_fp16} / 2**{k} -> term={term} (FP16).")

    return term


# ============================================================
# Core class
# ============================================================
<<<<<<< HEAD
=======
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

>>>>>>> orgin/main
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

<<<<<<< HEAD
    lat: int = 3

=======
>>>>>>> orgin/main
    def one_operation(self, kqed: np.float16) -> Tuple[int, np.float16, np.float16]:
        """
        One cycle update.

        Returns:
          (newMax, newRslt, exped)
        """
        newMax, newRslt, exped = self.eval(kqed)
        self.maxExp = int(newMax)
        self.sumDnm = np.float16(newRslt)
<<<<<<< HEAD
        self.update_ctrl()
        self.trgCnt += 1
        return int(newMax), np.float16(newRslt), np.float16(exped)
=======
        DmnValid = self.update_ctrl()
        self.trgCnt += 1
        return SMSMOutput(maxExp = int(newMax), sumDnm = np.float16(newRslt), exped = np.float16(exped), DmnValid = DmnValid, trgCnt = self.trgCnt)
>>>>>>> orgin/main

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
<<<<<<< HEAD
=======
        # print(exped)
>>>>>>> orgin/main

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
<<<<<<< HEAD
=======
            # Still validate result finiteness
>>>>>>> orgin/main
            if not np.isfinite(newRslt):
                raise NewRsltError(f"newRslt non-finite on exped==0 early path: {newRslt}")
            return newMax, newRslt, exped

        # exponent extraction (biased) -> unbiased candidate
        biased = fp16_biased_exp_field(exped)
<<<<<<< HEAD
=======
        # biased==0 would imply zero/subnormal (handled); biased==31 implies inf/nan (handled)
>>>>>>> orgin/main
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

<<<<<<< HEAD
        if (newMax > 15) or (newMax < -14):
            raise NewMaxError(f"newMax out of range [-14, 15]: newMax={newMax}")

=======
        # newMax guard
        if (newMax > 15) or (newMax < -14):
            raise NewMaxError(f"newMax out of range [-14, 15]: newMax={newMax}")

        # newRslt guard
>>>>>>> orgin/main
        if not np.isfinite(newRslt):
            raise NewRsltError(f"newRslt became non-finite: newRslt={newRslt}")

        return int(newMax), np.float16(newRslt), np.float16(exped)

    def update_ctrl(self) -> None:
        """Update colCnt and reset row state at end-of-row."""
        if self.colCnt == (self.seqLen - 1):
            self.colCnt = 0
            self.maxExp = -14
            self.sumDnm = np.float16(0.0)
<<<<<<< HEAD
        else:
            self.colCnt += 1


# ============================================================
# Spec checker (unchanged)
=======
            return 1
        else:
            self.colCnt += 1
            return 0


# ============================================================
# main: pattern checks on return values
>>>>>>> orgin/main
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
<<<<<<< HEAD
=======

    # ---- type checks ----
>>>>>>> orgin/main
    assert isinstance(newMax, int), "newMax must be int"
    assert isinstance(newRslt, np.float16), "newRslt must be fp16"
    assert isinstance(exped, np.float16), "exped must be fp16"

<<<<<<< HEAD
    assert -14 <= newMax <= 15, f"newMax out of range: {newMax}"

    assert np.isfinite(newRslt), f"newRslt not finite: {newRslt}"
    assert np.isfinite(exped) or exped == np.float16(0), f"exped invalid: {exped}"

=======
    # ---- exponent range ----
    assert -14 <= newMax <= 15, f"newMax out of range: {newMax}"

    # ---- finiteness ----
    assert np.isfinite(newRslt), f"newRslt not finite: {newRslt}"
    assert np.isfinite(exped) or exped == np.float16(0), f"exped invalid: {exped}"

    # ---- exped == 0 rule ----
>>>>>>> orgin/main
    if exped == np.float16(0) and colUsed != seqLen - 1:
        assert newMax == prevMax, "exped==0 must not update maxExp"
        assert newRslt == prevSum, "exped==0 must not update sumDnm"

<<<<<<< HEAD
=======
    # ---- column progression ----
>>>>>>> orgin/main
    if colUsed == seqLen - 1:
        assert colAfter == 0, "colCnt must reset at end of row"
    else:
        assert colAfter == colUsed + 1, "colCnt must increment by 1"

<<<<<<< HEAD

# ============================================================
# NEW: Golden math correctness test (separate from spec checker)
# ============================================================
def golden_row_math(
    row_inputs: List[np.float16],
) -> Tuple[int, np.float16]:
    """
    Golden (row-level) math model for the intended quantity:

        maxExpGolden = max unbiased exponent among exped_fp16 (normal-only), after subnormal flush-to-zero
        sumDnmGolden = FP16-accumulated sum of fp16(exped / 2**maxExpGolden) for nonzero exped

    Notes:
      - Uses the *same* exped generation semantics as SMSM.eval():
          exp in float32 -> quantize to fp16 -> flush subnormals to 0
      - Uses the *same* checked_scale_pow2_fp16() for scaling.
      - Accumulation is quantized to FP16 on every add.
    """
    exped_list: List[np.float16] = []
    max_exp_candidates: List[int] = []

    for x in row_inputs:
        x_fp16 = np.float16(x)
        exped = np.float16(np.exp(np.float32(x_fp16)))
        if is_fp16_subnormal(exped):
            exped = np.float16(0.0)
        if not np.isfinite(exped):
            raise ExponentialError(f"[golden] exped is non-finite after FP16 quantization: exped={exped}")

        exped_list.append(exped)

        if exped != np.float16(0.0):
            cand = fp16_unbiased_exp_if_normal(exped)
            # If it is nonzero but not normal, that would contradict our flush policy;
            # keep it strict:
            if cand is None:
                raise ExponentialError(f"[golden] Nonzero exped is not normal: exped={exped}")
            max_exp_candidates.append(cand)

    if len(max_exp_candidates) == 0:
        # Entire row exp underflowed / flushed -> spec's reset baseline for denom.
        return -14, np.float16(0.0)

    maxExpGolden = int(max(max_exp_candidates))

    sum_fp16 = np.float16(0.0)
    for exped in exped_list:
        if exped == np.float16(0.0):
            continue
        term = checked_scale_pow2_fp16(exped, maxExpGolden)
        sum_fp16 = np.float16(sum_fp16 + np.float16(term))

    if not np.isfinite(sum_fp16):
        raise NewRsltError(f"[golden] sumDnmGolden became non-finite: {sum_fp16}")

    return maxExpGolden, np.float16(sum_fp16)


def test_smsm_math_correctness(
    *,
    seed: int,
    numRows: int,
    seqLen: int,
    relTol: float = 1e-3,
) -> None:
    """
    Row-level math correctness test (separate from spec correctness):

    For each row:
      1) Drive SMSM for seqLen cycles
      2) Capture the *last-cycle* (newMax, newRslt) as the row output
      3) Compute (maxExpGolden, sumDnmGolden)
      4) Assert:
           newMax == maxExpGolden
           |newRslt - sumDnmGolden| / max(sumDnmGolden, tiny) < relTol

    Important:
      - This checks "is it the right number?", not just "did you follow the rules?"
    """
    rng = np.random.default_rng(seed)
    smsm = SMSM(seqLen=seqLen)

    for r in range(numRows):
        # Build one row. You can swap this distribution as you like.
        row_inputs = [np.float16(rng.normal(0.0, 1.0)) for _ in range(seqLen)]

        # Drive SMSM for one row; capture last-cycle outputs.
        last_newMax: int | None = None
        last_newRslt: np.float16 | None = None

        for i in range(seqLen):
            colUsed = smsm.colCnt
            prevMax = smsm.maxExp
            prevSum = smsm.sumDnm

            newMax, newRslt, exped = smsm.one_operation(row_inputs[i])

            # Keep your spec checker running (optional but recommended)
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

            if colUsed == seqLen - 1:
                last_newMax = newMax
                last_newRslt = newRslt

        assert last_newMax is not None and last_newRslt is not None, "Internal: failed to capture row output"

        gMax, gSum = golden_row_math(row_inputs)

        assert last_newMax == gMax, f"[row {r}] maxExp mismatch: hw={last_newMax}, golden={gMax}"

        denom = float(gSum) if float(gSum) != 0.0 else 1.0  # avoid div0; if golden==0, compare absolute via denom=1
        relErr = abs(float(last_newRslt) - float(gSum)) / denom
        assert relErr < relTol, (
            f"[row {r}] sumDnm mismatch: hw={float(last_newRslt):.6e}, "
            f"golden={float(gSum):.6e}, relErr={relErr:.3e}, tol={relTol:.3e}"
        )


# ============================================================
# main
# ============================================================
def main() -> None:
    np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

    # --- Existing "spec + trace" loop (keep if you want) ---
    # If you want to keep the long trace printing, you can leave your loop here.
    # Otherwise, comment it out to avoid massive logs.

    # --- NEW: math correctness test (separate) ---
    # Keep seqLen smaller for a fast unit test; set to 8192 if you really want full length.
    test_smsm_math_correctness(
        seed=0,
        numRows=3,
        seqLen=8192,      # change to 8192 for full-spec stress
        relTol=1e-3,
    )

    print("MATH CORRECTNESS TEST PASSED (row-level)")

if __name__ == "__main__":
    main()
=======
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
>>>>>>> orgin/main
