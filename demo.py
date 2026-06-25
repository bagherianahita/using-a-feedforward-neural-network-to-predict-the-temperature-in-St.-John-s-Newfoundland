"""CLI demo — St. John's temperature forecast."""

from forecast import train_and_forecast


def main() -> None:
    print("Temperature prediction demo — St. John's, NL (synthetic hourly data)\n")
    result = train_and_forecast()
    print("Next 6 hours forecast (°C):", [round(x, 2) for x in result["pred_c"]])
    print("Actual holdout (°C):       ", [round(x, 2) for x in result["actual_c"]])
    print(f"MAE (NN):        {result['mae_nn']:.2f}°C")
    print(f"MAE (baseline):  {result['mae_baseline']:.2f}°C")
    print("\nDemo complete — no API keys or external data required.")
    print("Browser UI: streamlit run app.py")


if __name__ == "__main__":
    main()
