"""Independent bit-packed GF(2) repair preserving reliability and likelihood ties."""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _bit(rows, row, col):
    return (rows[row, col >> 6] >> np.uint64(col & 63)) & np.uint64(1)


@njit(cache=True)
def osd_repair(h, syndrome, posterior, prior, max_order, search_width):
    """Repair over GF(2), preserving the original reliability order and ties.

    Matrix rows are packed into 64-bit words for elimination. Pivot selection,
    free-variable search, and floating-point likelihood sums retain their
    original order. Return (binary error, whether the full syndrome is valid).
    """
    m, n = h.shape
    if syndrome.size != m or posterior.size != n or prior.size != n:
        raise ValueError("matrix, syndrome, and LLR dimensions do not agree")
    hard = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        hard[j] = np.uint8(posterior[j] <= 0.0)
    order = np.argsort(np.abs(posterior))
    # Keep original column positions to pack contiguous input rows. The
    # reliability permutation governs pivot selection, never matrix storage.
    words = (n + 63) >> 6
    reduced = np.zeros((m, words), dtype=np.uint64)
    rhs = np.empty(m, dtype=np.uint8)
    for i in range(m):
        value = syndrome[i]
        for j in range(n):
            if h[i, j]:
                reduced[i, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
            value ^= h[i, j] & hard[j]
        rhs[i] = value
    pivots = np.empty(min(m, n), dtype=np.int64)
    is_pivot = np.zeros(n, dtype=np.uint8)
    rank = 0
    for col in range(n):
        if rank == m:
            break
        selected = -1
        for row in range(rank, m):
            if _bit(reduced, row, order[col]):
                selected = row
                break
        if selected < 0:
            continue
        if selected != rank:
            for word in range(words):
                tmp = reduced[rank, word]
                reduced[rank, word] = reduced[selected, word]
                reduced[selected, word] = tmp
            tmp_rhs = rhs[rank]
            rhs[rank] = rhs[selected]
            rhs[selected] = tmp_rhs
        for row in range(m):
            if row != rank and _bit(reduced, row, order[col]):
                for word in range(words):
                    reduced[row, word] ^= reduced[rank, word]
                rhs[row] ^= rhs[rank]
        pivots[rank] = col
        is_pivot[col] = 1
        rank += 1
    for row in range(rank, m):
        if rhs[row]:
            return hard, False
    result = hard.copy()
    for row in range(rank):
        if rhs[row]:
            result[order[pivots[row]]] ^= np.uint8(1)
    width = max(0, min(search_width, n - rank))
    free_cols = np.empty(width, dtype=np.int64)
    count = 0
    for col in range(n):
        if count == width:
            break
        if not is_pivot[col]:
            free_cols[count] = col
            count += 1
    pivot_cost = np.empty(rank, dtype=np.float64)
    for row in range(rank):
        original_col = order[pivots[row]]
        pivot_cost[row] = prior[original_col] * (1.0 - 2.0 * result[original_col])
    deltas = np.empty(width, dtype=np.float64)
    for a in range(width):
        col = free_cols[a]
        original_col = order[col]
        delta = prior[original_col] * (1.0 - 2.0 * result[original_col])
        for row in range(rank):
            if _bit(reduced, row, order[col]):
                delta += pivot_cost[row]
        deltas[a] = delta
    best_delta = 0.0
    best_a, best_b = -1, -1
    if max_order >= 1:
        for a in range(width):
            if deltas[a] < best_delta:
                best_delta, best_a = deltas[a], a
    if max_order >= 2:
        for a in range(width):
            col_a = free_cols[a]
            for b in range(a + 1, width):
                col_b = free_cols[b]
                delta = deltas[a] + deltas[b]
                for row in range(rank):
                    if _bit(reduced, row, order[col_a]) and _bit(reduced, row, order[col_b]):
                        delta -= 2.0 * pivot_cost[row]
                if delta < best_delta:
                    best_delta, best_a, best_b = delta, a, b
    for selection in range(2):
        chosen = best_a if selection == 0 else best_b
        if chosen >= 0:
            col = free_cols[chosen]
            result[order[col]] ^= np.uint8(1)
            for row in range(rank):
                if _bit(reduced, row, order[col]):
                    result[order[pivots[row]]] ^= np.uint8(1)
    for row in range(m):
        check = np.uint8(0)
        for col in range(n):
            check ^= h[row, col] & result[col]
        if check != syndrome[row]:
            return result, False
    return result, True

