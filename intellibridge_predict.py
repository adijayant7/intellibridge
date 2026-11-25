r"""IntelliBridge - bridge DANGER LEVEL (1-100) from sensor readings. Accepts ANY numeric value.

Run (cmd):
    cd /d C:\lumo
    python intellibridge_predict.py            # train first time, then ask for readings
    python intellibridge_predict.py --demo     # 5 example situations
    python intellibridge_predict.py --retrain  # force retraining

How it works (short):
 1. Physics step: turn the 8 raw readings into 7 "how abnormal" numbers (temperature-corrected,
    sign-blind: a closing crack is as bad as an opening one).
 2. ML step A: Isotonic Regression learns each sensor's own severity curve (always rising).
    ML step B: monotonic Gradient Boosting combines the 7 severities into one danger level 1-100. Monotonic = a worse reading can never lower the danger level.
 3. Level -> class:  1-29 Safe,  30-64 Warning,  65-100 Danger.
Training data is SIMULATED (no public dataset has all these sensors). Say "simulated" in review.
"""
import argparse, os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

FEATS = ['vib_rms_mg', 'dom_freq_hz', 'tilt_deg', 'crack_mm', 'water_cm', 'strain_ue', 'temp_c', 'humidity_pct']
PROMPTS = {
    'vib_rms_mg':   ('Vibration RMS from MPU6050', 'mg', 'normal 2.8 to 5, heavy truck 6 to 12'),
    'dom_freq_hz':  ('Dominant vibration frequency', 'Hz', 'healthy about 4.0, drops when stiffness is lost'),
    'tilt_deg':     ('Tilt from MPU6050', 'deg', 'normal under 0.2 (either direction)'),
    'crack_mm':     ('Crack gap change (ultrasonic)', 'mm', 'normal within +/-0.4 (opening or closing)'),
    'water_cm':     ('Water level above normal', 'cm', 'normal 0 to 60, flood above 100'),
    'strain_ue':    ('Strain gauge', 'microstrain', 'normal within +/-100 after temperature'),
    'temp_c':       ('Temperature', 'C', 'context only, does not raise danger by itself'),
    'humidity_pct': ('Humidity', '%', 'does not change structural danger'),
}
TYPICAL = dict(vib_rms_mg=3.0, dom_freq_hz=4.0, tilt_deg=.05, crack_mm=0.0, water_cm=20.0, strain_ue=0.0, temp_c=20.0, humidity_pct=65.0)
CLASSES = ['Safe', 'Warning', 'Danger']
ACTION = {'Safe': 'Normal operation. Keep the routine sampling schedule.',
          'Warning': 'Raise the sampling rate, queue an MQTT report, schedule an inspection.',
          'Danger': 'CLOSE THE SAFETY GATE, sound the local alarm, alert the engineers now.'}
def level_to_class(level): return 'Danger' if level >= 65 else ('Warning' if level >= 30 else 'Safe')

# ------------------------------------------------ step 1: physics (raw readings -> abnormality)
PF = ['vib', 'freq_drop', 'freq_rise', 'tilt', 'crack', 'water', 'strain']
PF_NAME = {'vib': 'vib_rms_mg', 'freq_drop': 'dom_freq_hz', 'freq_rise': 'dom_freq_hz', 'tilt': 'tilt_deg',
           'crack': 'crack_mm', 'water': 'water_cm', 'strain': 'strain_ue'}
def physical(d):
    """Raw readings -> 7 'how abnormal' numbers. Temperature must never raise danger by itself, so for
    frequency, crack and strain the milder of two readings counts: the raw value (gauge already temperature
    compensated) or the value after removing thermal drift (uncompensated gauge)."""
    T = d.temp_c; fc = d.dom_freq_hz + 0.015 * (T - 20)          # frequency corrected to 20 C
    return pd.DataFrame(dict(vib=d.vib_rms_mg.abs(),
        freq_drop=np.minimum(4.0 - d.dom_freq_hz, 4.0 - fc).clip(lower=0),
        freq_rise=np.minimum(d.dom_freq_hz - 4.5, fc - 4.5).clip(lower=0),
        tilt=d.tilt_deg.abs(),
        crack=np.minimum(d.crack_mm.abs(), (d.crack_mm - 0.02 * (T - 20)).abs()),
        water=d.water_cm.clip(lower=0),
        strain=np.minimum(d.strain_ue.abs(), (d.strain_ue - 6 * (T - 20)).abs())))

# engineering severity curve per sensor: (abnormality, danger 0-100). Increasing only.
CURVES = {
 'vib':       ([0, 8, 12, 20, 30, 50],          [0, 0, 30, 50, 70, 100]),
 'freq_drop': ([0, .1, .25, .5, .65, 1, 2],     [0, 0, 28, 45, 65, 85, 100]),
 'freq_rise': ([0, 1, 2, 6, 20, 46],            [0, 0, 35, 55, 70, 100]),
 'tilt':      ([0, .1, .3, .5, 1, 2.5, 5],      [0, 0, 30, 48, 66, 90, 100]),
 'crack':     ([0, .25, .4, .6, 1, 3, 6],       [0, 0, 30, 47, 66, 92, 100]),
 'water':     ([0, 60, 100, 140, 180, 260, 400], [0, 0, 30, 48, 66, 90, 100]),
 'strain':    ([0, 70, 100, 160, 200, 300, 600], [0, 0, 30, 50, 66, 85, 100]),
}
def danger_target(P):
    """Teaching target: worst sensor decides, other abnormal sensors add 15% of their severity."""
    S = np.column_stack([np.interp(P[k], *CURVES[k]) for k in PF]); mx = S.max(1)
    return np.minimum(100, mx + .15 * (S.sum(1) - mx))

# ------------------------------------------------ simulated training data (covers ANY magnitude)
NORMAL_HI = dict(vib=10, freq_drop=.15, freq_rise=1, tilt=.15, crack=.3, water=60, strain=80)
BIG = dict(vib=5000, freq_drop=60, freq_rise=5000, tilt=1000, crack=1000, water=5000, strain=1e5)
DANGER_AT = dict(vib=30, freq_drop=.65, freq_rise=6, tilt=1.0, crack=1.0, water=180, strain=200)
def simulate_phys(n, seed):
    rng = np.random.default_rng(seed); cols = {}
    for k in PF:
        normal = rng.uniform(0, NORMAL_HI[k], n)
        lo = NORMAL_HI[k] * .5
        mild = rng.uniform(lo, 1.5 * DANGER_AT[k], n)                                     # mild to dangerous (most of the learning)
        wide = rng.uniform(lo, 4 * DANGER_AT[k], n)
        huge = np.exp(rng.uniform(np.log(lo), np.log(BIG[k]), n))                        # absurd values up to huge numbers
        u = rng.random(n); abn = np.where(u < .6, mild, np.where(u < .85, wide, huge))
        cols[k] = np.where(rng.random(n) < .4, abn, normal)
    return pd.DataFrame(cols)

def fixed_threshold_rule(d):
    """Base-paper style: hard-coded limit per sensor."""
    danger = (d.vib_rms_mg > 12) | (d.tilt_deg.abs() > 1.0) | (d.crack_mm.abs() > 1.0) | (d.water_cm > 180) | (d.strain_ue.abs() > 200)
    warn = (d.vib_rms_mg > 6) | (d.tilt_deg.abs() > .4) | (d.crack_mm.abs() > .4) | (d.water_cm > 100) | (d.strain_ue.abs() > 100)
    return np.where(danger, 'Danger', np.where(warn, 'Warning', 'Safe'))

def _sev(bundle, P):
    """Stage A: per-sensor monotone severity 0-100 (isotonic regression learned from data)."""
    return np.column_stack([bundle['iso'][k].predict(P[k].values) for k in PF])
def _combine(bundle, S):
    """Stage B: monotone gradient boosting combines the 7 severities into one level."""
    mx = S.max(1); return np.clip(bundle['model'].predict(np.column_stack([mx, S.sum(1) - mx])), 1, 100)

def train(data_csv=None, model_path='intellibridge_model.joblib'):
    rng = np.random.default_rng(5)
    if data_csv:   # real data: columns = 8 readings + danger (0-100)
        d = pd.read_csv(data_csv).dropna(subset=FEATS + ['danger']); P = physical(d); y = d.danger.values; src = f'your data: {data_csv}'
        iso = {k: IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0, y_max=100).fit(P[k], y) for k in PF}
    else:
        src = 'SIMULATED data (120000 samples, any magnitude)'
        iso = {}
        for k in PF:        # Stage A data: one sensor abnormal, others normal -> its own severity curve
            x = np.concatenate([rng.uniform(0, 1.5 * DANGER_AT[k], 6000), rng.uniform(0, 4 * DANGER_AT[k], 3000),
                                np.exp(rng.uniform(np.log(NORMAL_HI[k] * .5), np.log(BIG[k]), 3000))])
            iso[k] = IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0, y_max=100).fit(x, np.interp(x, *CURVES[k]))
        P = simulate_phys(120000, 1); y = danger_target(P)
    Ptr, Pte, ytr, yte = train_test_split(P, y, test_size=.25, random_state=0)
    b = dict(iso=iso, source=src, safe_median=TYPICAL)
    S = _sev(b, Ptr); mx = S.max(1)                      # Stage B data
    b['model'] = HistGradientBoostingRegressor(monotonic_cst=[1, 1], max_iter=300, learning_rate=.1, random_state=0).fit(np.column_stack([mx, S.sum(1) - mx]), ytr)
    p = _combine(b, _sev(b, Pte))
    cls_ok = np.mean([level_to_class(a) == level_to_class(c) for a, c in zip(p, yte)])
    print(f'\nTRAINED on {src}; {len(Ptr)} train / {len(Pte)} test samples')
    print(f'Danger-level error: mean {np.mean(np.abs(p - yte)):.2f} points (scale 1-100), 95% of cases within {np.percentile(np.abs(p - yte), 95):.1f} points')
    print(f'Safe/Warning/Danger class agreement with engineering limits: {cls_ok:.3f}')
    joblib.dump(b, model_path); print(f'Saved {model_path}\n'); return joblib.load(model_path)

# ------------------------------------------------ prediction
def predict(bundle, reading):
    """Returns (class, level 1-100, [sensors that drive it])."""
    P = physical(pd.DataFrame([reading], columns=FEATS).astype(float))[PF]
    S = _sev(bundle, P); level = float(_combine(bundle, S)[0])
    why = []                                   # sensors whose own severity is clearly raised, worst first
    for i in np.argsort(-S[0]):
        f = PF_NAME[PF[i]]
        if S[0][i] >= 25 and f not in why: why.append(f)
    return level_to_class(level), int(round(level)), why[:3]

def show(bundle, reading, title=None):
    cls, level, why = predict(bundle, reading)
    if title: print(f'\n=== {title} ===')
    print('   ' + ', '.join(f'{k.split("_")[0]} {reading[k]:g}' for k in FEATS))
    print(f'   DANGER LEVEL: {level} / 100   ->   {cls.upper()}')
    if why: print('   Main reasons: ' + ', '.join(PROMPTS[f][0] for f in why))
    print('   Action: ' + ACTION[cls]); return cls

def ask(bundle):
    print('\nEnter the sensor readings (ANY number is accepted). Enter = typical value in [ ]. q = quit.')
    while True:
        r = {}
        for f in FEATS:
            name, unit, hint = PROMPTS[f]; default = TYPICAL[f]
            while True:
                s = input(f'{name} ({unit}) [{default}]  ({hint}): ').strip()
                if s.lower() == 'q': return
                try: r[f] = default if s == '' else float(s); assert np.isfinite(r[f]); break
                except (ValueError, AssertionError): print('   Please type a number.')
        show(bundle, r, 'PREDICTION')
        if input('\nAnother reading? (Enter = yes, n = no): ').strip().lower() == 'n': return

def demo(bundle):
    N = TYPICAL
    cases = [('Normal day', {}), ('Heavy truck on a hot day (NOT damage)', dict(vib_rms_mg=9.5, dom_freq_hz=3.9, crack_mm=.3, strain_ue=130, temp_c=38, humidity_pct=40)),
             ('Early structural damage', dict(vib_rms_mg=3.1, dom_freq_hz=3.5, tilt_deg=.4, crack_mm=.5, water_cm=15, strain_ue=90, temp_c=22)),
             ('Flood, structure fine', dict(water_cm=210, humidity_pct=90)),
             ('Severe damage', dict(vib_rms_mg=4, dom_freq_hz=3.1, tilt_deg=1.4, crack_mm=1.8, strain_ue=220))]
    for t, kw in cases:
        r = dict(N); r.update(kw); show(bundle, r, t)
        print('   Fixed-threshold rule would say: ' + fixed_threshold_rule(pd.DataFrame([r]))[0].upper())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--retrain', action='store_true'); ap.add_argument('--demo', action='store_true')
    ap.add_argument('--data'); ap.add_argument('--model', default='intellibridge_model.joblib'); a = ap.parse_args()
    if a.retrain or a.data or not os.path.exists(a.model): bundle = train(a.data, a.model)
    else: bundle = joblib.load(a.model); print(f'Loaded {a.model} (trained on {bundle["source"]})')
    demo(bundle) if a.demo else ask(bundle)
