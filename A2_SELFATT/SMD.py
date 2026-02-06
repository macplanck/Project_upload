from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


# =========================
# Errors (spec violations)
# =========================
class SmdError(Exception):
    """Base class for SMD-related errors."""


class InDmnError(SmdError):
    """Raised when inDmn is missing or non-finite at colCnt == 0."""


class InExpError(SmdError):
    """Raised when inExp is missing at colCnt == 0."""


class NewRowInfoError(SmdError):
    """Raised when inExp/inDmn are provided when colCnt != 0."""


class SmdDenominatorError(SmdError):
    """Raised when stored denominator smDmn is 0, subnormal, NaN, or Inf."""


class Q1ExpoError(SmdError):
    """Raised when q1 exponent field would exceed FP16 normal range (1..30)."""


class Fp16NumericError(SmdError):
    """Raised when q1 or newDiv becomes NaN/Inf."""


# ==================================
# FP16 helpers (bitfield operations)
# ==================================
FP16_EXP_BITS = 5
FP16_FRAC_BITS = 10
FP16_EXP_MAX = (1 << FP16_EXP_BITS) - 1  # 31
FP16_EXP_MAX_NORMAL = FP16_EXP_MAX - 1   # 30


def fp16_to_bits(val: np.float16) -> np.uint16:
    """Convert np.float16 to raw uint16 bits."""
    return np.frombuffer(np.float16(val).tobytes(), dtype=np.uint16)[0]


def bits_to_fp16(bits: int) -> np.float16:
    """Convert raw uint16 bits to np.float16."""
    return np.frombuffer(np.uint16(bits).tobytes(), dtype=np.float16)[0]


def fp16_fields(val: np.float16) -> Tuple[int, int, int]:
    """
    Extract (sign, exp_field, frac_field) from FP16.
    exp_field is biased exponent field in [0..31].
    frac_field is fraction in [0..1023].
    """
    bits = int(fp16_to_bits(val))
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    return sign, exp, frac


def is_subnormal_fp16(val: np.float16) -> bool:
    """Subnormal iff exp==0 and frac!=0."""
    _, exp, frac = fp16_fields(val)
    return (exp == 0) and (frac != 0)


def make_fp16_from_fields(sign: int, exp: int, frac: int) -> np.float16:
    """
    Pack FP16 from fields without any arithmetic.
    Note: caller must ensure exp/frac are in range.
    """
    bits = ((sign & 0x1) << 15) | ((exp & 0x1F) << 10) | (frac & 0x3FF)
    return bits_to_fp16(bits)


# =========================
# SMD block simulator
# =========================
@dataclass
class SMD:
    """
    Softmax Division (SMD) block simulator.

    Spec notes implemented:
    - All FP ops are conducted in float16 for arithmetic divisions.
    - Scaling by 2^smExp is implemented by FP16 exponent-field adjustment.
    - Subnormal results of q1/newDiv are flushed to zero (FTZ).
    - Underflow is not an error unless explicitly checked (we do not error on FTZ).
    """

    # Model parameter
    seqLen: int = 8192

    # Control/state
    colCnt: int = 0
    trgCnt: int = 0

    # Stored row parameters
    smExp: int = 0               # unbiased exponent used for scaling by 2^smExp
    smDmn: np.float16 = np.float16(1.0)

    # Latency (documentary only)
    lat: int = 2

    def one_operation(
        self,
        inNmn: np.ndarray,
        inExp: Optional[int] = None,
        inDmn: Optional[np.float16] = None,
    ) -> np.ndarray:
        """
        Process one 32-wide slice.

        Parameters
        ----------
        inNmn : np.ndarray
            Shape (32,), dtype float16. Numerators.
        inExp : Optional[int]
            Only valid when colCnt == 0; latches to self.smExp.
        inDmn : Optional[np.float16]
            Only valid when colCnt == 0; latches to self.smDmn.

        Returns
        -------
        np.ndarray
            Shape (32,), dtype float16. Divided results.

        Raises
        ------
        InDmnError, InExpError, NewRowInfoError, SmdDenominatorError,
        Q1ExpoError, Fp16NumericError
        """
        if not (isinstance(inNmn, np.ndarray) and inNmn.shape == (32,) and inNmn.dtype == np.float16):
            raise ValueError(f"inNmn must be np.ndarray(dtype=np.float16, shape=(32,)), got {inNmn.shape} in datatype {inNmn.dtype}")

        if self.colCnt == 0:
            if inDmn is None:
                raise InDmnError("inDmn is None at colCnt==0")
            if not np.isfinite(inDmn):
                raise InDmnError("inDmn is not finite at colCnt==0")
            if inExp is None:
                raise InExpError("inExp is None at colCnt==0")

            self.smExp = int(inExp)
            self.smDmn = np.float16(inDmn)
        # else:
        #     if (inExp is not None) or (inDmn is not None):
        #         raise NewRowInfoError("inExp/inDmn provided when colCnt != 0")

        new_div = self.eval(inNmn)
        self.update_ctrl()
        self.trgCnt += 1
        return new_div

    def eval(self, inNmn: np.ndarray) -> np.ndarray:
        """
        Evaluate q1 scaling by exponent adjustment, then divide by denominator.
        """
        # Denominator must not be 0/subnormal/NaN/Inf
        if (self.smDmn == np.float16(0.0)) or is_subnormal_fp16(self.smDmn) or (not np.isfinite(self.smDmn)):
            raise SmdDenominatorError("smDmn is 0/subnormal/NaN/Inf")

        out = np.empty((32,), dtype=np.float16)

        for i in range(32):
            x = np.float16(inNmn[i])
            sign, exp_field, frac_field = fp16_fields(x)

            # IMPORTANT:
            # The provided pseudocode assumes we can do:
            #   q1Expo = inNmn.exp_field - smExp
            # and then pack a normal number if q1Expo>0.
            #
            # This operation is only meaningful for NORMAL inputs (exp_field in 1..30).
            # For exp_field==0 (zero/subnormal) or 31 (Inf/NaN), we handle explicitly
            # to avoid generating nonsensical fields.

            if exp_field == 0:
                # input is zero or subnormal -> FTZ allowed; treat q1 as 0
                q1 = np.float16(0.0)

            elif exp_field == FP16_EXP_MAX:
                # input is Inf/NaN: propagate as-is, then will be caught by finite check
                q1 = x

            else:
                # normal input: exponent-field adjustment implements / 2^smExp
                q1_exp = int(exp_field) - int(self.smExp)

                if q1_exp > FP16_EXP_MAX_NORMAL:
                    raise Q1ExpoError(f"q1 exponent field {q1_exp} would exceed 30 (i={i})")

                if q1_exp > 0:
                    # pack as normal
                    q1 = make_fp16_from_fields(sign, q1_exp, frac_field)
                else:
                    # would be subnormal/zero -> FTZ
                    q1 = np.float16(0.0)

            # Flush q1 subnormal to zero (FTZ)
            if is_subnormal_fp16(q1):
                q1 = np.float16(0.0)

            # newDiv = q1 / smDmn (FP16 arithmetic)
            # Note: numpy will compute and then cast to float16 because q1 and smDmn are float16.
            new_div = np.float16(q1 / self.smDmn)

            # Flush newDiv subnormal to zero (FTZ)
            if is_subnormal_fp16(new_div):
                new_div = np.float16(0.0)

            # Numeric error checks
            if (not np.isfinite(q1)) or (not np.isfinite(new_div)):
                raise Fp16NumericError(f"q1 or newDiv became NaN/Inf at i={i}: q1={q1}, newDiv={new_div}")

            out[i] = new_div

        return out

    def update_ctrl(self) -> None:
        """Advance colCnt over 32-wide blocks."""
        max_col = (self.seqLen // 32) - 1
        if self.colCnt == max_col:
            self.colCnt = 0
        else:
            self.colCnt += 1


# =========================
# Golden model + test main
# =========================
def golden_smd_div(
    inNmn: np.ndarray,
    smExp: int,
    smDmn: np.float16,
) -> np.ndarray:
    """
    Golden model intended to match the spec:
    - Scaling by 2^smExp performed by exponent-field adjustment (normal inputs).
    - FTZ for q1/newDiv subnormals.
    - FP16 division for newDiv.

    This is deliberately the same behavior as SMD.eval, separated for testing.
    """
    smd = SMD()
    smd.smExp = int(smExp)
    smd.smDmn = np.float16(smDmn)
    return smd.eval(inNmn)


def main() -> None:
    np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

    rng = np.random.default_rng(0)
    smd = SMD(seqLen=8192)

    # --- Pattern: simulate one full row (seqLen/32 blocks) ---
    blocks = smd.seqLen // 32

    # Choose a safe exponent range to avoid frequent overflow in exponent-field adjustment.
    # (You can expand this for stress testing.)
    in_exp = int(rng.integers(low=-10, high=11))

    # Denominator must be finite and not 0/subnormal by spec.
    in_dmn = np.float16(1.25)

    for b in range(blocks):
        in_nmn = rng.normal(0.0, 1.0, size=(32,)).astype(np.float16)

        # Only provide new row info at colCnt==0
        if smd.colCnt == 0:
            hw = smd.one_operation(in_nmn, inExp=in_exp, inDmn=in_dmn)
        else:
            hw = smd.one_operation(in_nmn)

        gold = golden_smd_div(in_nmn, smd.smExp, smd.smDmn)

        # Value check (exact match is expected because the golden uses identical rules).
        if not np.array_equal(hw, gold):
            # Provide a helpful diff
            diff_idx = np.where(hw != gold)[0]
            raise AssertionError(
                f"Mismatch at block={b}, colCnt(after update)={smd.colCnt}, "
                f"indices={diff_idx.tolist()}, hw={hw[diff_idx]}, gold={gold[diff_idx]}"
            )

    # --- Negative tests: ensure spec-violation assertions fire ---
    # 1) Providing inExp/inDmn when colCnt != 0 should raise NewRowInfoError
    smd2 = SMD()
    x = rng.normal(0.0, 1.0, size=(32,)).astype(np.float16)
    smd2.one_operation(x, inExp=0, inDmn=np.float16(1.0))  # colCnt becomes 1
    try:
        smd2.one_operation(x, inExp=0, inDmn=np.float16(1.0))
        raise AssertionError("Expected NewRowInfoError was not raised.")
    except NewRowInfoError:
        pass

    # 2) Denominator subnormal should raise SmdDenominatorError
    smd3 = SMD()
    smd3.smExp = 0
    smd3.smDmn = np.float16(2.0 ** -24)  # subnormal in FP16
    try:
        smd3.eval(x)
        raise AssertionError("Expected SmdDenominatorError was not raised.")
    except SmdDenominatorError:
        pass

    print("All checks passed.")
    print(f"Final colCnt={smd.colCnt}, trgCnt={smd.trgCnt}, lat={smd.lat}")


if __name__ == "__main__":
    main()
