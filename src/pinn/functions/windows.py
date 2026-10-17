def split_windows(t_span: tuple[float, float], n: int) -> list[tuple[float, float]]:
    t0, tf = t_span
    edges = [t0 + (tf - t0) * k / n for k in range(n + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n)]
