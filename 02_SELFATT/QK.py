from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np


@dataclass
class PartialProductSim:
    """
    Simulates a 32-lane FP16 MAC datapath for attention-style dot products.

    Core behavior
    -------------
    - When q_flag==1: treat incoming 32 FP16 numbers as a Q tile and buffer it.
    - When q_flag==0: treat incoming 32 FP16 numbers as a K tile for (row_cnt, col_cnt),
      and if a Q tile is already buffered, output the partial dot-product for that tile.

    Model parameters
    ----------------
    hidden_size : int
        Total feature dimension. Must be a multiple of 32 for this simulator.
    seq_len : int
        Total sequence length (max row_cnt + 1).

    Quantization support
    --------------------
    If use_quant==True:
      - Inputs are assumed to represent quantized integers encoded in fp16 containers.
      - Dequant: x_deq = x_fp16 * scale
    This is intentionally simple and matches common HW pipelines (scale multiply later).

    Notes
    -----
    - This simulator is *tile-local*: it only computes 32-wide partial sums.
    - Accumulating across tiles to form a full dot-product is the caller's job.
    """

    hidden_size: int = 128
    seq_len: int = 8192
    use_quant: bool = True

    # Internal state
    _q_buf: Optional[np.ndarray] = None
    _q_row: Optional[int] = None
    _q_col_base: Optional[int] = None

    def step(
        self,
        vec32_fp16: Sequence[float] | np.ndarray,
        *,
        row_cnt: int,
        col_cnt: int,
        q_flag: int,
        qscale: float = 1.0,
        kscale: float = 1.0,
    ) -> Tuple[bool, np.float32]:
        """
        One cycle step.

        Parameters
        ----------
        vec32_fp16 : Sequence[float] | np.ndarray
            32 numbers representing either Q tile (if q_flag==1) or K tile (if q_flag==0).
            Values will be cast to np.float16.

        row_cnt : int
            Which row of K (0 <= row_cnt < seq_len).

        col_cnt : int
            Which column index of K we are working on (0 <= col_cnt < hidden_size).
            This simulator interprets col_cnt as belonging to a 32-wide tile:
              col_base = (col_cnt // 32) * 32

        q_flag : int
            1 => incoming is Q tile to buffer
            0 => incoming is K tile to MAC with buffered Q

        qscale : float
            Dequant scale for Q tile if use_quant==True.

        kscale : float
            Dequant scale for K tile if use_quant==True.

        Returns
        -------
        (out_valid, partial_product)
          out_valid : bool
              True only when q_flag==0 AND a Q tile was previously buffered for the same tile.
          partial_product : np.float32
              The 32-lane partial dot-product for that tile (0 if out_valid is False).
        """
        # ---- basic checks ----
        if self.hidden_size % 32 != 0:
            raise ValueError(f"hidden_size must be multiple of 32, got {self.hidden_size}")
        if not (0 <= row_cnt < self.seq_len):
            raise ValueError(f"row_cnt out of range: {row_cnt} (seq_len={self.seq_len})")
        if not (0 <= col_cnt < self.hidden_size):
            raise ValueError(f"col_cnt out of range: {col_cnt} (hidden_size={self.hidden_size})")

        v = np.asarray(vec32_fp16, dtype=np.float16)
        if v.shape != (32,):
            raise ValueError(f"vec32_fp16 must have shape (32,), got {v.shape}")

        col_base = (col_cnt // 32) * 32

        # ---- buffer Q tile ----
        if q_flag == 1:
            self._q_buf = v.copy()
            self._q_row = row_cnt          # optional: record when Q came in
            self._q_col_base = col_base    # bind Q tile to a specific hidden tile
            return (False, np.float32(0.0))

        # ---- compute with K tile ----
        if q_flag == 0:
            if self._q_buf is None:
                # Q not ready yet
                return (False, np.float32(0.0))

            if self._q_col_base != col_base:
                # We have a Q tile, but for a different 32-wide column tile
                # Caller may choose to buffer correct tile first.
                return (False, np.float32(0.0))

            # Dequant (optional)
            if self.use_quant:
                q = (self._q_buf.astype(np.float32) * np.float32(qscale))
                k = (v.astype(np.float32) * np.float32(kscale))
            else:
                q = self._q_buf.astype(np.float32)
                k = v.astype(np.float32)

            partial = np.dot(q, k).astype(np.float32)
            return (True, partial)

        raise ValueError(f"q_flag must be 0 or 1, got {q_flag}")
