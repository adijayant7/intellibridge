r"""Input handling around the IntelliBridge model. Accepts ANY number; only non-numbers are refused.
assess() never raises. The model is monotonic, so extreme values give a high danger level on their own."""
import numpy as np
import intellibridge_predict as ib

# Only used for a "check this sensor" note, never to reject a value.
USUAL = {'vib_rms_mg': (0, 200), 'dom_freq_hz': (0.5, 30), 'tilt_deg': (-30, 30), 'crack_mm': (-30, 30),
         'water_cm': (0, 800), 'strain_ue': (-3000, 3000), 'temp_c': (-40, 60), 'humidity_pct': (0, 100)}

def assess(bundle, reading):
    errors = []
    for k in ib.FEATS:
        v = reading.get(k)
        try: ok = v is not None and np.isfinite(float(v))
        except (TypeError, ValueError): ok = False
        if not ok: errors.append(f"{ib.PROMPTS[k][0]}: enter a number")
    if errors: return dict(status='invalid', errors=errors)
    reading = {k: float(reading[k]) for k in ib.FEATS}
    notes = []
    for k, (lo, hi) in USUAL.items():
        if not lo <= reading[k] <= hi:
            notes.append(f"{ib.PROMPTS[k][0]} = {reading[k]:g} is beyond what a real sensor normally reports ({lo:g} to {hi:g}); check the sensor. The danger level is still calculated.")
    cls, level, why = ib.predict(bundle, reading)
    return dict(status='ok', cls=cls, level=level, proba=None, why=[ib.PROMPTS[f][0] for f in why], rule_reasons=[],
                notes=notes, action=ib.ACTION[cls], overridden=False)

def load(path='intellibridge_model.joblib'):
    import os, joblib, warnings
    b = None
    if os.path.exists(path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error'); b = joblib.load(path)
            if 'safe_median' not in b or 'model' not in b or 'iso' not in b: b = None
        except Exception as e:
            print("Saved model does not match this PC / version (%s). Retraining now (about 10 s)..." % type(e).__name__); b = None
        if b is None:
            try: os.remove(path)
            except OSError: pass
    return b if b is not None else ib.train(None, path)
