import joblib
import streamlit as st

from risk import risk_band
from who_ranges import who_flags
from wqi import compute_wqi, wqi_category

st.set_page_config(page_title="Water Quality Risk Predictor", page_icon="\U0001F4A7", layout="centered")

# Shared colors for risk.py bands + wqi.py categories (they overlap on Excellent/Good/Poor/Very Poor).
BAND_COLOR = {
    "Excellent": "green", "Good": "green", "Fair": "yellow",
    "Poor": "orange", "Very Poor": "red", "Unsuitable": "red",
}


@st.cache_resource
def load_model():
    # model.joblib is ~200MB (TabPFN bundles its transformer weights) -- cache so Streamlit
    # doesn't reload it from disk on every widget interaction.
    return joblib.load("model.joblib")


bundle = load_model()
model, threshold, columns, feature_importance, wqi_model = (
    bundle["model"], bundle["threshold"], bundle["columns"], bundle["feature_importance"], bundle["wqi_model"]
)

st.title("\U0001F4A7 Water Quality Risk Predictor")
st.caption("Enter water test values to estimate potability risk.")

# Dataset medians, so the form is usable out of the box without typing all 9 values first.
defaults = {
    "ph": 7.0, "Hardness": 197.0, "Solids": 20927.0, "Chloramines": 7.1,
    "Sulfate": 333.0, "Conductivity": 421.0, "Organic_carbon": 14.2,
    "Trihalomethanes": 66.4, "Turbidity": 3.96,
}

with st.container(border=True):
    values = []
    cols = st.columns(3)
    for i, name in enumerate(columns):
        with cols[i % 3]:
            values.append(st.number_input(name, value=defaults[name]))
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    sample = dict(zip(columns, values))
    proba = model.predict_proba([values])[0, 1]
    label = "Potable" if proba >= threshold else "Not Potable"
    band = risk_band(proba)

    with st.container(border=True):
        st.subheader("ML Prediction")
        st.badge(label, color="green" if label == "Potable" else "red", icon="\U0001F4A7")
        st.progress(float(proba), text=f"{proba:.0%} confidence")
        st.caption("Risk band")
        st.badge(band, color=BAND_COLOR[band])

    with st.container(border=True):
        st.subheader("Rule-based cross-check")
        st.caption("WHO/EPA guideline ranges, approximate")
        flags = who_flags(sample)
        if flags:
            st.error(f"Outside typical safe range: {', '.join(flags)}")
        else:
            st.success("All parameters within typical safe ranges")

    with st.container(border=True):
        st.subheader("Water Quality Index")
        true_wqi = compute_wqi(sample)
        ml_wqi = wqi_model.predict([values])[0]
        wqi_col1, wqi_col2 = st.columns(2)
        with wqi_col1:
            st.metric("Computed WQI", f"{true_wqi:.1f}")
            st.badge(wqi_category(true_wqi), color=BAND_COLOR[wqi_category(true_wqi)])
        with wqi_col2:
            st.metric("ML-predicted WQI", f"{ml_wqi:.1f}")
            st.badge(wqi_category(ml_wqi), color=BAND_COLOR[wqi_category(ml_wqi)])
        st.caption(
            "The ML model reconstructs this index from raw readings alone (no formula at "
            "inference time) -- how closely the two numbers match is a live accuracy check."
        )

    with st.expander("What drives this model's predictions?"):
        st.bar_chart(dict(sorted(feature_importance.items(), key=lambda kv: -kv[1])[:5]))
