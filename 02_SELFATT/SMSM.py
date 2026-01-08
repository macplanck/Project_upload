from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np


def fp16_unbiased_exponent(val_fp16: np.float16) -> int:
    """
    Extract the FP16 unbiased exponent.

    Notes
    -----
    IEEE-754 binary16 layout:
      sign: 1 bit
      exp : 5 bits (bias = 15)
      frac: 10 bits

    For exp == 0 (zero/subnormal), this function returns -15 as a sentinel,
    matching the user's design choice that maxExp initializes to -15 (minimum).
    """
    bits = np.frombuffer(np.float16(val_fp16).tobytes(), dtype=np.uint16)[0]
    exp_field = (bits >> 10) & 0x1F  # 5-bit exponent field

    if exp_field == 0:
        return -15  # sentinel for zero/subnormal in this model

    return int(exp_field) - 15

@dataclass
class SMSM:
    """
    SMSM: Streaming Max/Sum (denominator) Manager inspired by Online Softmax.

    Tracks a per-row running maximum exponent (maxExp) and a compensated running sum (sumDnm),
    while iterating across columns of K (row-wise softmax).
    """

    # Control signal
    colCnt: int = 0  # which col of K we are working on (0..hiddenSize/headNum-1)

    # Cycle counter
    trgCnt: int = 0  # increments per one_operation()

    # Model parameters
    hiddenSize: int = 4096
    headNum: int = 32

    # Default latency (kept as a parameter for scheduling models)
    lat: int = 3

    # Storing unit
    maxExp: int = -15  # running maximum unbiased exponent of exp(kqed)
    sumDnm: float = 0.0  # sum(exp(inputs))/2^(maxExp)

    # Input (latest)
    kqed: np.float16 = np.float16(0.0)

    def one_operation(self, kqed: np.float16) -> Tuple[int, float]:
        """
        Perform one streaming update with one FP16 input kqed.

        Flow (per pseudocode)
        --------------------------
        newMax, newRslt = eval()
        self.maxExp = newMax
        self.sumDnm = newRslt
        update_ctrl()
        trgCnt ++
        return newMax, newRslt
        """
        self.kqed = np.float16(kqed)

        newMax, newRslt = self.eval()

        self.maxExp = newMax
        self.sumDnm = newRslt

        self.update_ctrl()
        self.trgCnt += 1

        return newMax, newRslt

    def eval(self) -> Tuple[int, float]:
        """
        Streaming evaluation step.

        Pseudocode mapping
        ------------------
        exped = e^kqed

        if (maxExp < (exped.exponent-15)):
            newRslt = (sumDnm + exped/(2^(maxExp))) / 2^(exped.exponent - maxExp - 15)
            newMax  = exped.exponent - 15
        else:
            newRslt = sumDnm + exped/(2^(maxExp))
            newMax  = maxExp
        """
        # Compute exp in float32 then quantize to FP16 (to mimic HW FP16 exp output)
        exped_fp16 = np.float16(np.exp(np.float32(self.kqed)))
        exped_unbiased = fp16_unbiased_exponent(exped_fp16)

        # Convert exped to float32 for accumulator math
        exped_f32 = float(np.float32(exped_fp16))

        # Base scaled contribution: exped / 2^(maxExp)
        # (matches your internal sumDnm definition)
        scaled_contrib = exped_f32 / (2.0 ** self.maxExp)

        if self.maxExp < exped_unbiased:
            # Rescale old sum to new max:
            # division factor = 2^(unbiasedExp - maxExp)
            scale = 2.0 ** (exped_unbiased - self.maxExp)
            newRslt = (self.sumDnm + scaled_contrib) / scale
            newMax = exped_unbiased
        else:
            newRslt = self.sumDnm + scaled_contrib
            newMax = self.maxExp

        return int(newMax), float(newRslt)

    def update_ctrl(self) -> None:
        """
        Update column control and reset row accumulators on row boundary.

        if (colCnt == hiddenSize/headNum - 1):
            colCnt = 0
            maxExp = -15
            sumDnm = 0
        else:
            colCnt++
        """
        lastCol = (self.hiddenSize // self.headNum) - 1

        if self.colCnt == lastCol:
            self.colCnt = 0
            self.maxExp = -15
            self.sumDnm = 0.0
        else:
            self.colCnt += 1


# pattern verification
def golden_softmax_denom(row_vals):
    exp_vals = np.exp(np.array(row_vals, dtype=np.float32))
    max_val = np.max(exp_vals)
    denom = np.sum(exp_vals / (2.0 ** math.floor(math.log2(max_val))))
    max_exp = math.floor(math.log2(max_val))
    return max_exp, denom

if __name__ == "__main__":
    def pattern_check(seed):
        rng = np.random.default_rng(seed)
        smsm = SMSM()

        rowWidth = smsm.hiddenSize // smsm.headNum

        all_rows = []

        # Generate inputs for two rows
        for row in range(2):
            row_inputs = []
            for _ in range(rowWidth):
                kqed = np.float16(rng.normal(0.0, 1.0))
                row_inputs.append(kqed)
            all_rows.append(row_inputs)

        print("=== PATTERN CHECK (2 ROWS) ===")

        # Run SMSM row by row
        for row_idx, row_inputs in enumerate(all_rows):
            for kqed in row_inputs:
                newMax, newSum = smsm.one_operation(kqed)

            # Golden for this row
            gMax, gSum = golden_softmax_denom(row_inputs)

            print(f"[Row {row_idx}]")
            print(f"  HW maxExp = {newMax}, Golden maxExp = {gMax}")
            print(f"  HW sumDnm = {newSum:.6e}, Golden sumDnm = {gSum:.6e}")

            # Assertion per row
            assert abs(newSum - gSum) / gSum < 1e-3, (
                f"DENOM MISMATCH on row {row_idx}"
            )

        print("PASS: two-row pattern check")


    for i in range(100):
        pattern_check(i)
