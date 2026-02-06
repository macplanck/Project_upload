from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


class SpecViolationError(AssertionError):
    """Raised when the simulator detects a spec violation (control/data/protocol)."""


@dataclass(frozen=True)
class QKOutput:
    """
    Output of one_operation().

    Fields
    ------
    prtSum : np.float16
        The updated accumulator value for the current (sliceCnt, colCnt).
        NOTE: This value is only a *final* score[q,k] when outValid == True.
    outValid : bool
        True only when the output is a final score element (final slice and K-cycle).
    kIndex : int
        Which K token index this output corresponds to (colCnt at the time of eval).
    sliceIndex : int
        Which slice index this output corresponds to (sliceCnt at the time of eval).
    qFlag : bool
        The qFlag at the time of eval (True means the cycle was Q-load).
    trgCnt : int
        Cycle counter after increment (i.e., count of one_operation calls so far).
    """
    prtSum: np.float16
    outValid: bool
    kIndex: int
    sliceIndex: int
    qFlag: bool
    trgCnt: int


def fp16_scalar(x: float | np.floating) -> np.float16:
    """
    Quantize a scalar to FP16.

    Parameters
    ----------
    x : float or np.floating
        Input value.

    Returns
    -------
    np.float16
        FP16-quantized value.
    """
    return np.float16(x)


def tree_sum_fp16(vec_fp16: np.ndarray) -> np.float16:
    """
    Pairwise (tree) reduce in FP16.

    This approximates hardware-style reduction better than naive sequential sum
    (still a software model; exact HW rounding may differ depending on pipeline).

    Parameters
    ----------
    vec_fp16 : np.ndarray
        1-D float16 array.

    Returns
    -------
    np.float16
        FP16 reduction result.
    """
    assert vec_fp16.dtype == np.float16 and vec_fp16.ndim == 1
    work = vec_fp16.copy()
    while work.size > 1:
        if work.size % 2 == 1:
            work = np.concatenate([work, np.array([np.float16(0.0)], dtype=np.float16)])
        a = work[0::2]
        b = work[1::2]
        work = np.array([fp16_scalar(float(ai) + float(bi)) for ai, bi in zip(a, b)], dtype=np.float16)
    return work[0]


class QK:
    """
    QK block simulator (Q-stationary attention-score row generation).

    Spec summary
    ------------
    - Process head-dim in 32-wide slices. For each slice:
      * 1 cycle: load Q slice (qFlag=1)
      * seqLen cycles: stream K tokens (qFlag=0), accumulate accu[k] += dot32(Q, K)
    - One score[q,k] element is FINAL only after the final slice for that k.
      We expose this as outValid in QKOutput.

    FP16 policy
    -----------
    - All stored values and arithmetic results are quantized to float16.
    - The shift factor 2**dqFac is treated as a Python float (not fp16), per your reminder.
    - Dequantization is modeled as: deq[i] = fp16( fp16(inNum[i]) / (2**dqFac) )
      (division happens in Python float then quantized to fp16; NumPy promotion rules
       make true fp16 division tricky without custom kernels).

    Control checker
    ---------------
    - Internal assertions detect spec violations:
      * illegal dqFac range for signed 5-bit
      * wrong qFlag/colCnt transitions
      * unexpected accu write on Q cycles (guarded)
    """

    def __init__(
        self,
        hiddenSize: int = 4096,
        headNum: int = 32,
        seqLen: int = 8192,
<<<<<<< HEAD
        lat: int = 4,
=======
        lat: int = 7,
>>>>>>> orgin/main
    ) -> None:
        # Floating-point exception behavior (as requested).
        np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

        # Model parameters
        self.hiddenSize = int(hiddenSize)
        self.headNum = int(headNum)
        self.seqLen = int(seqLen)
        self.lat = int(lat)
<<<<<<< HEAD
=======
        # latency:
        # {q or k}_buf  -> {q or k}_deq -> cos and sin for RoPE -> addsub and mult from RoPE -> QKmult -> add32to8 -> add8to2 -> add2toAccu
>>>>>>> orgin/main

        # Derived
        if self.headNum <= 0:
            raise ValueError("headNum must be positive.")
        if self.hiddenSize % self.headNum != 0:
            raise ValueError("hiddenSize must be divisible by headNum.")
        self.headDim = self.hiddenSize // self.headNum  # Dh
        if self.headDim % 32 != 0:
            raise ValueError("headDim must be divisible by 32 for 32-wide slicing.")
        self.sliceMax = (self.headDim // 32) - 1

        # Control signals (state)
        self.sliceCnt = 0
        self.colCnt = 0
        self.qFlag = True  # initialized high

        # Cycle counter
        self.trgCnt = 0

        # Storage
        self.qDeq = np.zeros((32,), dtype=np.float16)
        self.kDeq = np.zeros((32,), dtype=np.float16)
        self.accu = np.zeros((self.seqLen,), dtype=np.float16)

        # For control-checking
        self._prevSliceCnt = self.sliceCnt
        self._prevColCnt = self.colCnt
        self._prevQFlag = self.qFlag

    @staticmethod
    def _check_dqfac_signed_5bit(dqFac: int) -> None:
        """
        Enforce signed 5-bit integer range: [-16, 15].
        """
        if not (-16 <= int(dqFac) <= 15):
            raise SpecViolationError(f"dqFac out of signed-5bit range: {dqFac} (expected -16..15)")

    def _dequantize(self, inNum: np.ndarray, dqFac: int) -> np.ndarray:
        """
        Dequantize int8 vector -> fp16 vector.

        Spec: deq = fp16(fp16(inNum)/2**dqFac)
        Shift factor 2**dqFac is NOT fp16 materialized.

        Returns
        -------
        np.ndarray (32,), dtype float16
        """
        self._check_dqfac_signed_5bit(dqFac)

        if inNum.shape != (32,):
            raise SpecViolationError(f"inNum must have shape (32,), got {inNum.shape}")
        if inNum.dtype != np.int8:
            raise SpecViolationError(f"inNum must be int8, got {inNum.dtype}")

        scale = float(2.0 ** int(dqFac))  # shift amount as Python float (not fp16)

        # Elementwise quantization policy:
        # fp16(fp16(inNum[i]) / scale)
        out = np.empty((32,), dtype=np.float16)
        for i in range(32):
            num_fp16 = fp16_scalar(int(inNum[i]))
            out[i] = fp16_scalar(float(num_fp16) / scale)

        return out
<<<<<<< HEAD

    def eval(self, inNum: np.ndarray, dqFac: int) -> Tuple[np.float16, bool]:
=======
    
    def _rotate_half(self,x: np.ndarray) -> np.ndarray:
        """
        Rotate the last dimension by 90 degrees in even–odd pairs.

        For each pair (x[2i], x[2i+1]):
            -> (-x[2i+1], x[2i])

        Parameters
        ----------
        x : np.ndarray
            1D array with even length (e.g., head_dim).

        Returns
        -------
        np.ndarray
            Rotated array of the same shape as x.
        """
        assert x.ndim == 1, "rotate_half expects a 1D array"
        assert x.shape[0] % 2 == 0, "Length must be even"

        y = np.empty_like(x)
        y[0::2] = -x[1::2]
        y[1::2] =  x[0::2]
        return y

    def eval(self, inNum: np.ndarray, dqFac: int, ROPEcos: np.ndarray, ROPEsin: np.ndarray) -> Tuple[np.float16, bool]:
>>>>>>> orgin/main
        """
        Evaluate one cycle: either load Q slice or accumulate K slice.

        Returns
        -------
        (partialSum, outValid)
        """
        deq = self._dequantize(inNum, dqFac)

<<<<<<< HEAD
        if self.qFlag:
            # Q-load cycle
            self.qDeq = deq
=======
        deq_prime = self._rotate_half(deq)
        roped = []
        for i in range (len(deq)//2):
            for j in range(2):
                roped.append(deq[i*2 + j]*ROPEcos[i] + deq_prime[i*2 + j]*ROPEsin[i])

        if self.qFlag:
            # Q-load cycle
            self.qDeq = np.array(roped)
>>>>>>> orgin/main
            partialSum = self.accu[self.colCnt]  # should not be treated as valid output
            outValid = False
        else:
            # K cycle
<<<<<<< HEAD
            self.kDeq = deq
=======
            self.kDeq = np.array(roped)
>>>>>>> orgin/main
            # dot32 in FP16 (multiply then tree-sum)
            mul_fp16 = np.array(
                [fp16_scalar(float(self.kDeq[i]) * float(self.qDeq[i])) for i in range(32)],
                dtype=np.float16,
            )
            dot_fp16 = tree_sum_fp16(mul_fp16)
            partialSum = fp16_scalar(float(dot_fp16) + float(self.accu[self.colCnt]))

            # "one score[q,k] element is done only after the final slice for that k"
            outValid = (self.sliceCnt == self.sliceMax)

        return partialSum, outValid

    def update_ctrl(self) -> None:
        """
        Update control per spec.

        Spec:
        if colCnt == seqLen-1:
            qFlag=1; colCnt=0
            if sliceCnt==sliceMax:
                sliceCnt=0; accu[:]=0
            else:
                sliceCnt += 1
        elif not qFlag:
            colCnt += 1
        elif qFlag:
            qFlag = 0
        """
        if self.colCnt == (self.seqLen - 1):
            self.qFlag = True
            self.colCnt = 0

            if self.sliceCnt == self.sliceMax:
                self.sliceCnt = 0
                self.accu[:] = np.float16(0.0)
            else:
                self.sliceCnt += 1

        elif (not self.qFlag):
            self.colCnt += 1

        elif self.qFlag:
            self.qFlag = False

    def _assert_control_invariants(self) -> None:
        """
        Checker to assert spec violations in control evolution.

        This runs AFTER update_ctrl() to verify that the state transition matches
        one of the allowed patterns.
        """
        ps = self._prevSliceCnt
        pc = self._prevColCnt
        pq = self._prevQFlag

        ns = self.sliceCnt
        nc = self.colCnt
        nq = self.qFlag

        # Allowed transitions, based on previous state:
        # Case A: previous was Q-load (pq=True) -> must go to K mode, same indices
        if pq is True:
            if not (nq is False and ns == ps and nc == pc):
                raise SpecViolationError(
                    f"Control violation after Q-load: "
                    f"(sliceCnt,colCnt,qFlag)=({ps},{pc},{pq}) -> ({ns},{nc},{nq}) "
                    f"expected qFlag to drop to 0 with same slice/col."
                )

        # Case B: previous was K cycle (pq=False)
        else:
            if pc == self.seqLen - 1:
                # End of K sweep; must reset col, set qFlag, advance slice or wrap+clear
                if not (nq is True and nc == 0 and (ns == ps + 1 or (ps == self.sliceMax and ns == 0))):
                    raise SpecViolationError(
                        f"Control violation at end-of-row-sweep: "
                        f"(sliceCnt,colCnt,qFlag)=({ps},{pc},{pq}) -> ({ns},{nc},{nq}) "
                        f"expected (qFlag=1,colCnt=0) and slice advance/wrap."
                    )
            else:
                # Normal K streaming: colCnt increments by 1, others unchanged
                if not (nq is False and ns == ps and nc == pc + 1):
                    raise SpecViolationError(
                        f"Control violation during K-stream: "
                        f"(sliceCnt,colCnt,qFlag)=({ps},{pc},{pq}) -> ({ns},{nc},{nq}) "
                        f"expected colCnt increment by 1."
                    )

<<<<<<< HEAD
    def one_operation(self, inNum: np.ndarray, dqFac: int) -> QKOutput:
=======
    def one_operation(self, inNum: np.ndarray, dqFac: int, ROPEcos: np.ndarray, ROPEsin: np.ndarray) -> QKOutput:
>>>>>>> orgin/main
        """
        Perform one cycle of operation.

        Parameters
        ----------
        inNum : np.ndarray
            Shape (32,), dtype int8. The quantized input vector for this cycle
            (either Q slice when qFlag=1, or K slice when qFlag=0).
        dqFac : int
            Signed 5-bit integer in [-16, 15]. Dequant exponent.
<<<<<<< HEAD
=======
        ROPEcos: np.ndarray
            Shape (16,), dtype float16
        ROPEsin: np.ndarray
            Shape (16,), dtype float16
>>>>>>> orgin/main

        Returns
        -------
        QKOutput
            Includes prtSum, outValid, indices, and control snapshot.

        Notes
        -----
        - We only WRITE accu[colCnt] on K cycles (qFlag==0). Writing on Q cycles
          is a common spec-risk, so we guard it here.
        """
        # Snapshot indices for the output (state at time of eval)
        kIndex = self.colCnt
        sliceIndex = self.sliceCnt
        qFlagNow = self.qFlag

<<<<<<< HEAD
        prtSum, outValid = self.eval(inNum, dqFac)
=======
        prtSum, outValid = self.eval(inNum, dqFac, ROPEcos, ROPEsin)
>>>>>>> orgin/main

        # Accu write: only on K cycles
        if not qFlagNow:
            self.accu[kIndex] = prtSum

        # Update control and increment cycle counter
        self._prevSliceCnt, self._prevColCnt, self._prevQFlag = self.sliceCnt, self.colCnt, self.qFlag
        self.update_ctrl()
        self._assert_control_invariants()

        self.trgCnt += 1

        return QKOutput(
            prtSum=prtSum,
            outValid=bool(outValid and (not qFlagNow)),  # final slice AND K-cycle
            kIndex=int(kIndex),
            sliceIndex=int(sliceIndex),
            qFlag=bool(qFlagNow),
            trgCnt=int(self.trgCnt),
        )


def golden_score_row(
    qSlices: List[np.ndarray],
    kSlicesPerToken: List[List[np.ndarray]],
    dqFacQ: int,
    dqFacK: int,
) -> np.ndarray:
    """
    Compute golden attention-score row in a straightforward way (FP16-quantized ops).

    Parameters
    ----------
    qSlices : list of np.ndarray
        Length = sliceCount. Each element shape (32,), dtype int8.
    kSlicesPerToken : list of list of np.ndarray
        Outer length = seqLen. Inner length = sliceCount. Each slice shape (32,), dtype int8.
    dqFacQ : int
        Dequant exponent for Q.
    dqFacK : int
        Dequant exponent for K.

    Returns
    -------
    np.ndarray
        Shape (seqLen,), dtype float16. Full attention-score row.
    """
    np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

    sliceCount = len(qSlices)
    seqLen = len(kSlicesPerToken)
    out = np.zeros((seqLen,), dtype=np.float16)

    def deq_vec(x: np.ndarray, dq: int) -> np.ndarray:
        scale = float(2.0 ** int(dq))
        y = np.empty((32,), dtype=np.float16)
        for i in range(32):
            y[i] = fp16_scalar(float(fp16_scalar(int(x[i]))) / scale)
        return y

    qDeqSlices = [deq_vec(qSlices[s], dqFacQ) for s in range(sliceCount)]

    for k in range(seqLen):
        acc = np.float16(0.0)
        for s in range(sliceCount):
            kDeq = deq_vec(kSlicesPerToken[k][s], dqFacK)
            mul_fp16 = np.array(
                [fp16_scalar(float(kDeq[i]) * float(qDeqSlices[s][i])) for i in range(32)],
                dtype=np.float16,
            )
            dot_fp16 = tree_sum_fp16(mul_fp16)
            acc = fp16_scalar(float(acc) + float(dot_fp16))
        out[k] = acc

    return out


def pattern_check() -> None:
    """
    Minimal pattern to check QK simulator correctness on a small configuration.

    What it checks
    --------------
    1) Control-schedule invariants (via SpecViolationError assertions inside QK)
    2) Numerical agreement vs a golden model for final outputs (outValid cycles only)
    """
    rng = np.random.default_rng(123)

    # Use a small config for sanity test
    hiddenSize = 128
    headNum = 1
    seqLen = 16

    qk = QK(hiddenSize=hiddenSize, headNum=headNum, seqLen=seqLen, lat=4)

    sliceCount = (qk.headDim // 32)
    assert sliceCount == 4, "With hiddenSize=128, headNum=1, expect 4 slices of 32."

    dqFacQ = 3
    dqFacK = 3

    # Generate one Q row worth of slices and K slices per token
    qSlices: List[np.ndarray] = []
    for _ in range(sliceCount):
        qSlices.append(rng.integers(-128, 127, size=(32,), dtype=np.int8))

    kSlicesPerToken: List[List[np.ndarray]] = []
    for _k in range(seqLen):
        slices = []
        for _s in range(sliceCount):
            slices.append(rng.integers(-128, 127, size=(32,), dtype=np.int8))
        kSlicesPerToken.append(slices)

    # Drive the simulator: for each slice: 1 Q cycle then seqLen K cycles
    collected_final = np.zeros((seqLen,), dtype=np.float16)
    collected_mask = np.zeros((seqLen,), dtype=bool)

    for s in range(sliceCount):
        # Q-load cycle
        outQ = qk.one_operation(qSlices[s], dqFacQ)
        assert outQ.outValid is False, "Q cycle must not be outValid."

        # K stream
        for k in range(seqLen):
            outK = qk.one_operation(kSlicesPerToken[k][s], dqFacK)

            # Only final slice outputs should be valid
            if s == sliceCount - 1:
                assert outK.outValid is True, "Final slice K cycles must be outValid."
                collected_final[outK.kIndex] = outK.prtSum
                collected_mask[outK.kIndex] = True
            else:
                assert outK.outValid is False, "Non-final slice K cycles must not be outValid."

    # Ensure we collected all seqLen final outputs
    assert collected_mask.all(), f"Did not collect all final outputs: mask={collected_mask}"

    # Golden comparison
    golden = golden_score_row(qSlices, kSlicesPerToken, dqFacQ, dqFacK)

    # FP16 is coarse; exact equality is plausible here because both use same fp16 quantization steps.
    # Still, we allow a small tolerance.
    diff = np.abs(collected_final.astype(np.float32) - golden.astype(np.float32))
    maxDiff = float(diff.max())

    print("=== PATTERN CHECK ===")
    print(f"max |sim - golden| = {maxDiff:.6e}")
    print("sample outputs (k=0..5):")
    for i in range(min(6, seqLen)):
        print(f"k={i:2d} sim={float(collected_final[i]): .6e} golden={float(golden[i]): .6e} diff={float(diff[i]): .6e}")

    assert maxDiff <= 5e-3, f"Numerical mismatch too large: maxDiff={maxDiff}"


if __name__ == "__main__":
    pattern_check()
