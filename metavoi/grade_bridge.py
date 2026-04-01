GRADE_THRESHOLDS = [
    (0.05, "High", "Low EVPI. Additional trials unlikely to change the decision."),
    (0.20, "Moderate", "Moderate EVPI. Consider a targeted confirmatory trial."),
    (0.40, "Low", "High EVPI. Strong case for new trial — decision is uncertain."),
    (1.01, "Very Low", "Very high EVPI. Decision is highly uncertain — new evidence essential."),
]


def grade_from_p_wrong(p_wrong):
    """Map probability of wrong decision to GRADE-like certainty."""
    for threshold, certainty, recommendation in GRADE_THRESHOLDS:
        if p_wrong <= threshold:
            return {
                "certainty": certainty,
                "p_wrong": p_wrong,
                "recommendation": recommendation,
            }
    return {
        "certainty": "Very Low",
        "p_wrong": p_wrong,
        "recommendation": GRADE_THRESHOLDS[-1][2],
    }
