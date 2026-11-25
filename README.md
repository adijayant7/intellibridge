# IntelliBridge

A low-cost, offline multi-sensor framework for real-time bridge health risk classification.
An ESP32 reads vibration/tilt (MPU6050), crack gap (ultrasonic), water level, strain, temperature and
humidity, turns them into a **danger level from 1 to 100** (Safe / Warning / Danger) and drives an offline
safety gate. MQTT is used only for non-critical reporting.

**Team:** Yash Raj Singh, Manish Patel, N. Swathi, Aditya Jayant, Yashodhara Singh, Likitha Vijayakumar

## What is in this repo
| Path | What it is |
|---|---|
| `index.html` | The dashboard. One file, runs the model in the browser. No server, no Python. |
| `streamlit_app.py` | The same live predictor as a Streamlit app, using the real Python model. |
| `src/` | Python: LUMO feature extraction, vibration model, multi-sensor danger model, tests, ESP32 exporter. |
| `esp32/` | The model as C tables (`intellibridge_danger.h`) and a sketch that runs in the Wokwi simulator. |
| `docs/` | Project presentation. |

## Run it
- **Dashboard:** double-click `index.html`.
- **Streamlit:** `pip install -r requirements.txt` then `streamlit run streamlit_app.py`
- **Tests:** `cd src` then `python test_bridge_cases.py`
- **ESP32 export:** `cd src` then `python export_esp32.py`

## Deploy
- **Vercel:** import this GitHub repo, Framework Preset **Other**, leave build command and output directory empty. It serves `index.html`.
- **GitHub Pages:** Settings, Pages, deploy from branch `main`, folder `/ (root)`.
- **Streamlit Community Cloud:** New app, pick this repo, main file `streamlit_app.py`. The first visit trains the model (about 20 s).

## Models
1. **Vibration model (real data).** 24 spectral features per 60 s window from the LUMO bridge-tower data, Mahalanobis anomaly score trained on healthy windows, tested on groups it never saw (mean file AUC 0.91 against 0.39 for a fixed RMS threshold).
2. **Multi-sensor danger model.** Physics step (temperature correction, sign-blind) then isotonic regression per sensor, then monotonic gradient boosting to one 1-100 level. It accepts any value, and a worse reading never lowers the level.

## Honest limits
The multi-sensor model is trained on **simulated** data because no public dataset has all five sensors; its danger limits are engineering assumptions. The LUMO test has 5 healthy and 5 damaged files per group from one tower, with simulated sensor noise. The ESP32 does not yet compute features from live MPU6050 readings. The LUMO dataset itself is not included in this repo.
