""" CUDA Parallel Monotonic Alignment Search. """

import math
import warnings

from numba import cuda, NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@cuda.jit('(float32[:,:,:], float32[:,:,:], int16[:,:,:], int16[:,:,:], int16[:], int16[:])')
def cuda_b_mas(log_p, prev_log_p, prev_ind, attn_out, in_lens, out_lens):
    bx, jx = cuda.grid(2)
    tpb, tpj = cuda.gridsize(2)

    if bx >= log_p.shape[0]:
        return

    for b in range(bx, log_p.shape[0], tpb):
        n, m = out_lens[b], in_lens[b]
        if jx == 0:
            log_p[b, 0, 1:m] = -math.inf
        cuda.syncthreads()

        for i in range(1, n):
            for j in range(jx, m, tpj):  # for each text dim
                if j > 0 and log_p[b, i - 1, j - 1] >= log_p[b, i - 1, j]:
                    prev_log_p[b, i, j] = log_p[b, i - 1, j - 1]
                    prev_ind[b, i, j] = j - 1
                else:
                    prev_log_p[b, i, j] = log_p[b, i - 1, j]
                    prev_ind[b, i, j] = j

            cuda.syncthreads()

            for j in range(jx, m, tpj):
                log_p[b, i, j] += prev_log_p[b, i, j]

            cuda.syncthreads()

        # backtrack
        if jx == 0:
            curr_text_idx = m - 1
            for i in range(n - 1, -1, -1):
                attn_out[b, i, curr_text_idx] = 1
                curr_text_idx = prev_ind[b, i, curr_text_idx]
