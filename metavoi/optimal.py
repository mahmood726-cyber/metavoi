from metavoi.evsi import compute_evsi_curve


def find_optimal_n(inp, n_values=None):
    """Find trial size N that maximizes EVSI_pop - cost."""
    curve = compute_evsi_curve(inp, n_values=n_values)

    if not curve:
        return {
            "optimal_n": None,
            "optimal_evsi_pop": 0,
            "optimal_net_benefit": 0,
            "breakeven_n": None,
            "curve": [],
        }

    best = max(curve, key=lambda pt: pt.net_benefit)
    optimal_n = best.n if best.net_benefit > 0 else None

    breakeven_n = find_breakeven_n(curve)

    return {
        "optimal_n": optimal_n,
        "optimal_evsi_pop": best.evsi_pop,
        "optimal_net_benefit": best.net_benefit,
        "breakeven_n": breakeven_n,
        "curve": curve,
    }


def find_breakeven_n(curve):
    """Find largest N where EVSI_pop still exceeds cost."""
    breakeven = None
    for pt in curve:
        if pt.net_benefit >= 0:
            breakeven = pt.n
    return breakeven
