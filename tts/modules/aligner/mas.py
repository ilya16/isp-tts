""" Parallel Monotonic Alignment Search. """

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def mas_width1(log_p):
    """mas with hardcoded width=1"""
    # assumes mel x text
    log_p[0, 1:] = -np.inf
    log_p[:, 0] = np.cumsum(log_p[:, 0])
    for i in range(1, log_p.shape[0]):
        log_p[i, 1:] += np.maximum(log_p[i - 1, :-1], log_p[i - 1, 1:])  # for each text dim

    prev_ind = np.zeros_like(log_p, dtype=np.int16)
    prev_ind[1:, 1:] = np.arange(1, log_p.shape[1], 1) - (log_p[:-1, :-1] >= log_p[:-1, 1:])

    # backtrack
    opt = np.zeros_like(log_p, dtype=np.int16)
    curr_text_idx = log_p.shape[1] - 1
    for i in range(log_p.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]

    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens):
    attn_out = np.zeros_like(b_attn_map, dtype=np.int16)
    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, :out_lens[b], :in_lens[b]])
        attn_out[b, :out_lens[b], :in_lens[b]] = out
    return attn_out
