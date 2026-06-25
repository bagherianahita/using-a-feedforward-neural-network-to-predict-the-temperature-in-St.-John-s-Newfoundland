"""Streamlit demo — St. John's temperature forecast (feedforward neural network)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from forecast import train_and_forecast

DEFAULT_PORT = 8507

st.set_page_config(page_title="St. John's Temperature Forecast", page_icon="🌡️", layout="wide")
st.title("St. John's Temperature Forecast")
st.caption(
    "Feedforward neural network predicting the next 6 hours from synthetic hourly data "
    "(no API keys or downloads required)."
)

with st.sidebar:
    st.header("Settings")
    hours = st.slider("Training hours", 200, 800, 500, step=50)
    lookback = st.slider("Lookback window (hours)", 6, 24, 12)
    run = st.button("Run forecast", type="primary", use_container_width=True)

if "result" not in st.session_state or run:
    with st.spinner("Training feedforward NN on synthetic St. John's data…"):
        st.session_state.result = train_and_forecast(hours=hours, lookback=lookback)

result = st.session_state.result

col1, col2, col3 = st.columns(3)
col1.metric("NN MAE", f"{result['mae_nn']:.2f} °C")
col2.metric("Persistence baseline MAE", f"{result['mae_baseline']:.2f} °C")
delta = result["mae_baseline"] - result["mae_nn"]
col3.metric("NN vs baseline", f"{delta:+.2f} °C", help="Positive = NN beats persistence")

st.subheader("6-hour forecast (°C)")
forecast_df = pd.DataFrame(
    {
        "Hour ahead": [f"+{i + 1}h" for i in range(len(result["pred_c"]))],
        "Neural network": result["pred_c"].round(2),
        "Actual (holdout)": result["actual_c"].round(2),
        "Persistence baseline": result["baseline_c"].round(2),
    }
)
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

st.subheader("Recent temperature series")
chart_df = pd.DataFrame({"hour": result["history_hours"], "temp_c": result["history_temp"]})
st.line_chart(chart_df.set_index("hour"))

st.divider()
st.markdown(
    f"**Run locally:** `streamlit run app.py` → http://localhost:{DEFAULT_PORT}  \n"
    "**CLI:** `python demo.py`"
)
