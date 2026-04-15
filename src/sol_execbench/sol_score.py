def sol_score(
    t_k: float,
    t_b: float,
    t_sol: float,
) -> float:
    """
    Compute anchored score S(T_k).

    S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))

    T_b can be set as any fast implementation of the reference solution. T_b is set as an optimized PyTorch implementation of the reference solution for the SOL-ExecBench dataset.

    Args:
        t_k: Candidate kernel runtime (ms)
        t_b: Optimized scoring baseline runtime (ms)
        t_sol: Speed-of-Light runtime (ms), raw SOLAR estimate

    Returns:
        Score in [0, 1]. S=0.5 when T_k=T_b, S=1 when T_k=T_SOL.
    """
    denom_gap = t_b - t_sol
    if denom_gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / denom_gap)
