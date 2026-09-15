"""Map a model's potability probability to a human-readable risk band."""

BANDS = [
    (0.80, "Excellent"),
    (0.60, "Good"),
    (0.40, "Fair"),
    (0.20, "Poor"),
    (0.0, "Very Poor"),
]


def risk_band(potable_probability: float) -> str:
    for threshold, label in BANDS:
        if potable_probability >= threshold:
            return label
    return "Very Poor"


if __name__ == "__main__":
    assert risk_band(0.95) == "Excellent"
    assert risk_band(0.5) == "Fair"
    assert risk_band(0.05) == "Very Poor"
    print("risk.py self-check passed")
