r"""IntelliBridge prediction app - type bridge readings in a web form, get the risk.

Run (cmd):
    cd /d C:\lumo
    python intellibridge_app.py
A browser tab opens at http://localhost:8000 . Press Ctrl+C in cmd to stop.

Needs intellibridge_predict.py in the same folder (it trains the model the first time).
"""
import json, os, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import joblib, pandas as pd
import intellibridge_predict as ib
import intellibridge_safety as sf

MODEL = 'intellibridge_model.joblib'
bundle = sf.load(MODEL)
safe = {k: round(float(v), 2) for k, v in bundle['safe_median'].items()}

FIELDS = [dict(key=k, label=ib.PROMPTS[k][0], unit=ib.PROMPTS[k][1], hint=ib.PROMPTS[k][2], default=safe[k]) for k in ib.FEATS]
PRESETS = {
  'Normal day': dict(vib_rms_mg=3.0, dom_freq_hz=4.0, tilt_deg=.05, crack_mm=0, water_cm=20, strain_ue=0, temp_c=20, humidity_pct=65),
  'Heavy truck, hot day': dict(vib_rms_mg=9.5, dom_freq_hz=3.9, tilt_deg=.05, crack_mm=.3, water_cm=20, strain_ue=130, temp_c=38, humidity_pct=40),
  'Early damage': dict(vib_rms_mg=3.1, dom_freq_hz=3.5, tilt_deg=.4, crack_mm=.5, water_cm=15, strain_ue=90, temp_c=22, humidity_pct=65),
  'Flood': dict(vib_rms_mg=3.0, dom_freq_hz=4.0, tilt_deg=.05, crack_mm=0, water_cm=210, strain_ue=0, temp_c=20, humidity_pct=90),
  'Severe damage': dict(vib_rms_mg=4.0, dom_freq_hz=3.1, tilt_deg=1.4, crack_mm=1.8, water_cm=20, strain_ue=220, temp_c=20, humidity_pct=65),
}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IntelliBridge Predictor</title><style>
:root{--bg:#f5f4f0;--card:#fff;--ink:#272525;--mut:#6b6a73;--line:#dcdbe6;--lav:#dadbf1;--safe:#1f8a4c;--warn:#c27a00;--dng:#c62828}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 16px}h1{margin:0 0 4px;font-size:28px}p.sub{margin:0 0 20px;color:var(--mut)}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:20px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}.presets button{background:var(--lav);border:0;border-radius:999px;padding:7px 13px;font:inherit;font-size:13px;cursor:pointer}
.presets button:hover{filter:brightness(.95)}
.row{display:grid;grid-template-columns:1fr 120px;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid #eeedf5}
.row label{font-weight:600;font-size:14px}.row small{display:block;color:var(--mut);font-weight:400;font-size:12px}
.row input{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;font:inherit;text-align:right}
.go{margin-top:16px;width:100%;padding:13px;border:0;border-radius:10px;background:var(--ink);color:#fff;font:inherit;font-weight:700;font-size:16px;cursor:pointer}
.badge{font-size:40px;font-weight:800;margin:4px 0 2px}.Safe{color:var(--safe)}.Warning{color:var(--warn)}.Danger{color:var(--dng)}
.bar{height:10px;background:#eeedf5;border-radius:6px;overflow:hidden;margin:3px 0 10px}.bar i{display:block;height:100%}
.pl{display:flex;justify-content:space-between;font-size:13px}.act{margin-top:12px;padding:12px;border-radius:10px;background:var(--lav);font-size:14px}
.cmp{margin-top:12px;font-size:13px;color:var(--mut)}.empty{color:var(--mut)}.err{color:var(--dng);font-size:13px;margin-top:8px}
</style></head><body><div class="wrap"><h1>IntelliBridge Predictor</h1>
<p class="sub">Enter the bridge sensor readings and the trained model predicts Safe, Warning or Danger.</p>
<div class="grid"><div class="card"><div class="presets" id="presets"></div><div id="form"></div>
<button class="go" id="go">Predict bridge condition</button><div class="err" id="err"></div></div>
<div class="card" id="out"><p class="empty">The prediction will appear here. Pick an example above or type your own readings, then press Predict.</p></div></div>
<p class="sub" style="margin-top:16px;font-size:12px">Model: __SRC__. Simulated training data: shows the method, not real-world accuracy. Any number is accepted; danger level rises with every worse reading.</p></div>
<script>
const FIELDS=__FIELDS__,PRESETS=__PRESETS__,COL={Safe:'#1f8a4c',Warning:'#c27a00',Danger:'#c62828'};
const form=document.getElementById('form');
FIELDS.forEach(f=>{form.insertAdjacentHTML('beforeend',`<div class="row"><label for="${f.key}">${f.label} (${f.unit})<small>${f.hint}</small></label><input id="${f.key}" type="number" step="any" value="${f.default}"></div>`)});
const pr=document.getElementById('presets');
Object.keys(PRESETS).forEach(n=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>{FIELDS.forEach(f=>document.getElementById(f.key).value=PRESETS[n][f.key]);run()};pr.appendChild(b)});
async function run(){const err=document.getElementById('err');err.textContent='';const d={};
for(const f of FIELDS){const v=parseFloat(document.getElementById(f.key).value);if(isNaN(v)){err.textContent='Please enter a number for: '+f.label;return}d[f.key]=v}
const r=await(await fetch('/predict',{method:'POST',body:JSON.stringify(d)})).json();
const out=document.getElementById('out');
if(r.status==='invalid'){out.innerHTML=`<div class="badge" style="font-size:26px;color:#c62828">CHECK YOUR INPUT</div><p>No prediction was made:</p><ul>${r.errors.map(e=>`<li>${e}</li>`).join('')}</ul>`;return}
const lv=r.level,col=COL[r.cls];
const gauge=`<div class="pl"><span>Danger level</span><span><b style="font-size:26px;color:${col}">${lv}</b> / 100</span></div><div class="bar" style="height:16px;position:relative"><i style="width:${lv}%;background:${col}"></i></div><div class="pl" style="font-size:11px;color:#6b6a73"><span>1-29 Safe</span><span>30-64 Warning</span><span>65-100 Danger</span></div>`;
const nt=r.notes.length?`<div class="act" style="background:#fff4d6"><b>Check:</b><ul style="margin:4px 0 0 18px;padding:0">${r.notes.map(n=>`<li>${n}</li>`).join('')}</ul></div>`:'';
out.innerHTML=`<div class="empty">Predicted condition</div><div class="badge ${r.cls}">${r.cls.toUpperCase()}</div>${gauge}
${r.why.length?`<div style="margin-top:10px"><b>Main reasons:</b> ${r.why.join(', ')}</div>`:''}<div class="act"><b>Action:</b> ${r.action}</div>${nt}
<div class="cmp">A fixed-threshold rule (base-paper style) would say: <b>${r.rule.toUpperCase()}</b></div>`}
document.getElementById('go').onclick=run;
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype):
        b = body.encode(); self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        page = PAGE.replace('__FIELDS__', json.dumps(FIELDS)).replace('__PRESETS__', json.dumps(PRESETS)).replace('__SRC__', bundle['source'])
        self._send(200, page, 'text/html; charset=utf-8')
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        try: reading = {k: float(d[k]) for k in ib.FEATS}
        except (KeyError, TypeError, ValueError): reading = {k: float('nan') for k in ib.FEATS}
        out = sf.assess(bundle, reading)
        if out['status'] == 'ok': out['rule'] = str(ib.fixed_threshold_rule(pd.DataFrame([reading]))[0])
        self._send(200, json.dumps(out), 'application/json')

if __name__ == '__main__':
    port = 8000
    while True:
        try: srv = ThreadingHTTPServer(('127.0.0.1', port), H); break
        except OSError: port += 1
    url = f'http://localhost:{port}'
    print(f'IntelliBridge app running at {url}  (Ctrl+C to stop)')
    if '--no-browser' not in sys.argv: threading.Timer(1, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print('stopped')
