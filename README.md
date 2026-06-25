# Temperature Prediction — St. John's, NL

**Feedforward neural network** to predict the next 6 hours of temperature in St. John's, Newfoundland, compared against a persistence baseline.
<img width="1775" height="795" alt="image" src="https://github.com/user-attachments/assets/e2e74371-5955-4dce-af4e-a5dd470e44ae" />

![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)

---

## Architecture

```
┌──────────────┐   synthetic   ┌─────────────┐   scale    ┌──────────────┐
│ Hourly temps │ ────────────► │  Pandas     │ ─────────► │ MinMaxScaler │
│ (demo data)  │               │  windows    │            └──────┬───────┘
└──────────────┘               └─────────────┘                   │
                                                                 ▼
                        ┌─────────────────────────────────────────────┐
                        │  MLPRegressor (feedforward NN) vs baseline    │
                        └─────────────────────────────────────────────┘
```

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Browser demo trains on synthetic St. John's–style hourly data and shows a 6-hour forecast chart.

| | URL |
|---|-----|
| **Web UI** | http://localhost:8507 |
| **CLI demo** | `python demo.py` |

> Uses scikit-learn `MLPRegressor` (feedforward NN) — no TensorFlow download required for the demo.

---

## License

MIT — see [LICENSE](LICENSE).
