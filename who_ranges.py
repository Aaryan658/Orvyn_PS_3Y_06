"""Approximate WHO/EPA drinking-water guideline ranges, for a rule-based cross-check
next to the ML prediction. Values are commonly-cited guideline ballparks, not exact
regulatory text -- labelled as such in the UI.
"""

SAFE_RANGES = {
    "ph": (6.5, 8.5),
    "Hardness": (0, 300),
    "Solids": (0, 1000),
    "Chloramines": (0, 4),
    "Sulfate": (0, 250),
    "Conductivity": (0, 400),
    "Organic_carbon": (0, 4),
    "Trihalomethanes": (0, 80),
    "Turbidity": (0, 5),
}


def who_flags(values: dict) -> list[str]:
    """Return names of parameters outside the typical safe range."""
    return [name for name, v in values.items() if not (SAFE_RANGES[name][0] <= v <= SAFE_RANGES[name][1])]


if __name__ == "__main__":
    assert who_flags({"ph": 7.0, **{k: v[0] for k, v in SAFE_RANGES.items() if k != "ph"}}) == []
    assert who_flags({"ph": 14.0, **{k: v[0] for k, v in SAFE_RANGES.items() if k != "ph"}}) == ["ph"]
    print("who_ranges.py self-check passed")
