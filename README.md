# Temperature Prediction — St. John's, NL

**Feedforward neural network** to predict the next 6 hours of temperature in St. John's, Newfoundland, compared against a persistence baseline.

![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

---

## Architecture

```
┌──────────────┐   netCDF    ┌─────────────┐   scale    ┌──────────────┐
│ Environment  │ ──────────► │  Pandas     │ ─────────► │ MinMaxScaler │
│ Canada data  │             │  features   │            └──────┬───────┘
└──────────────┘             └─────────────┘                   │
                                                                 ▼
                        ┌─────────────────────────────────────────────┐
                        │  Feedforward NN  vs  persistence baseline   │
                        └─────────────────────────────────────────────┘
```

---

## Quick start (employers — ~2 min, synthetic data)

```bash
pip install -r requirements.txt
python demo.py
```

---

## License

MIT — see [LICENSE](LICENSE).
