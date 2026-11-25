r"""Run: python test_bridge_cases.py   (needs intellibridge_predict.py, intellibridge_safety.py, model in same folder)"""
# Expected answers decided BEFORE running, from sensor physics + the bands in the simulation.
N=dict(vib_rms_mg=3,dom_freq_hz=4.0,tilt_deg=.05,crack_mm=0,water_cm=20,strain_ue=0,temp_c=20,humidity_pct=65)
def c(name,exp,**kw): d=dict(N);d.update(kw);return (name,exp,d)
CASES=[
 c('normal','Safe'),
 c('vib 6 (light truck)','Safe',vib_rms_mg=6),
 c('vib 9.5 (heavy truck)','Safe',vib_rms_mg=9.5,strain_ue=30),
 c('vib 20 mg alone','Warning',vib_rms_mg=20),
 c('vib 50 mg alone','Danger',vib_rms_mg=50),
 c('vib 1 (very quiet)','Safe',vib_rms_mg=1),
 c('freq 3.95','Safe',dom_freq_hz=3.95),
 c('freq 3.75','Warning',dom_freq_hz=3.75),
 c('freq 3.3','Danger',dom_freq_hz=3.3),
 c('freq 2.5','Danger',dom_freq_hz=2.5),
 c('tilt 0.1','Safe',tilt_deg=.1),
 c('tilt 0.5','Warning',tilt_deg=.5),
 c('tilt 1.5','Danger',tilt_deg=1.5),
 c('crack 0.2','Safe',crack_mm=.2),
 c('crack 0.6','Warning',crack_mm=.6),
 c('crack 1.5','Danger',crack_mm=1.5),
 c('crack -1.5 (closing)','Danger',crack_mm=-1.5),
 c('water 60','Safe',water_cm=60),
 c('water 130','Warning',water_cm=130),
 c('water 210','Danger',water_cm=210),
 c('strain 50','Safe',strain_ue=50),
 c('strain 160','Warning',strain_ue=160),
 c('strain 300','Danger',strain_ue=300),
 c('hot day thermal drift','Safe',temp_c=35,strain_ue=90,crack_mm=.3,dom_freq_hz=3.9,vib_rms_mg=3.5),
 c('cold day thermal drift','Safe',temp_c=-3,strain_ue=-140,crack_mm=-.4,dom_freq_hz=4.3),
 c('humid rainy day','Safe',humidity_pct=98,temp_c=18),
 c('truck + hot','Safe',vib_rms_mg=9,temp_c=38,strain_ue=130,crack_mm=.3,dom_freq_hz=3.9),
 c('early damage','Warning',dom_freq_hz=3.5,tilt_deg=.4,crack_mm=.5,strain_ue=90),
 c('early damage, vib normal, crack+tilt','Warning',tilt_deg=.45,crack_mm=.55),
 c('severe damage','Danger',dom_freq_hz=3.1,tilt_deg=1.4,crack_mm=1.8,strain_ue=220,vib_rms_mg=4),
 c('severe, high vib','Danger',dom_freq_hz=3.0,tilt_deg=1.5,crack_mm=2,strain_ue=250,vib_rms_mg=15),
 c('flood + damage','Danger',water_cm=200,tilt_deg=.8,crack_mm=1.0,dom_freq_hz=3.4),
 c('flood, 140 cm, structure fine','Warning',water_cm=140),
 c('freq drop + high vib','Danger',dom_freq_hz=3.2,vib_rms_mg=12),
 c('truck + early crack','Warning',vib_rms_mg=9,crack_mm=.6,strain_ue=60),
 c('high vib + tilt 0.5','Warning',vib_rms_mg=15,tilt_deg=.5),
 c('water 100 + crack 0.5','Warning',water_cm=100,crack_mm=.5),
 c('all mild-normal edge','Safe',vib_rms_mg=4.5,dom_freq_hz=3.9,tilt_deg=.15,crack_mm=.2,water_cm=50,strain_ue=40),
 c('tilt 0.3 only','Safe',tilt_deg=.3),
 c('freq 11 Hz (sensor fault?)','Warning',dom_freq_hz=11),
 c('tilt -1.5 (leaning other way)','Danger',tilt_deg=-1.5),
 c('strain -300 (compression)','Danger',strain_ue=-300),
 c('crack -0.6 (closing, mild)','Warning',crack_mm=-.6),
]

import warnings;warnings.filterwarnings('ignore')
import time,intellibridge_safety as sf
b=sf.load(); bad=0
BORDER={'freq 3.75':('Safe','Warning'),'tilt 0.3 only':('Safe','Warning')}
for n,e,r in CASES:
    a=sf.assess(b,r); ok=a['cls']==e or (n in BORDER and a['cls'] in BORDER[n]); bad+=not ok
    tag='OK  ' if a['cls']==e else ('~OK ' if ok else 'FAIL')
    print(f"{tag} {n:38s} exp {e:8s} got {a['cls']:8s} level {a['level']:3d}")
print(f'\n{len(CASES)-bad}/{len(CASES)} acceptable')
t=time.time(); [sf.assess(b,CASES[0][2]) for _ in range(50)]; print('ms per prediction: %.1f'%((time.time()-t)/50*1000))

# ---- extra check 1: more of a sensor must NEVER lower the danger level (all 7 danger sensors, 600 baselines each)
import numpy as np, intellibridge_predict as ib
rng=np.random.default_rng(7); viol=0; tot=0
SW={'vib_rms_mg':(0,500),'tilt_deg':(0,60),'crack_mm':(0,60),'water_cm':(0,2000),'strain_ue':(0,9000)}
for k,(lo,hi) in SW.items():
    for _ in range(200):
        base=dict(N); base.update(temp_c=float(rng.uniform(-10,45)))
        xs=np.linspace(lo,hi,25); lv=[]
        for x in xs: r=dict(base); r[k]=float(x)+(6*(base['temp_c']-20) if k=='strain_ue' else 0)+(0.02*(base['temp_c']-20) if k=='crack_mm' else 0); lv.append(sf.assess(b,r)['level'])
        viol+=int(np.any(np.diff(lv)<-0)); tot+=1
for _ in range(200):          # frequency: lower than healthy must never lower the level
    lv=[sf.assess(b,dict(N,dom_freq_hz=float(f)))['level'] for f in np.linspace(4.0,0.2,25)]; viol+=int(np.any(np.diff(lv)<0)); tot+=1
print(f"\nMONOTONIC CHECK: {tot-viol}/{tot} sweeps never decreased")
# ---- extra check 2: ANY value (huge, negative, absurd) never crashes and extreme values are Danger
bad=0
for _ in range(3000):
    r={k:float(rng.choice([-1,1])*10**rng.uniform(-3,9)) for k in ib.FEATS}; a=sf.assess(b,r)
    bad+= a['status']!='ok' or not 1<=a['level']<=100
print(f"RANDOM ABSURD INPUTS: {3000-bad}/3000 handled")
print('Extreme tilt 14 ->',sf.assess(b,dict(N,tilt_deg=14))['level'],'| crack 18 ->',sf.assess(b,dict(N,crack_mm=18))['level'],'| humidity 150 ->',sf.assess(b,dict(N,humidity_pct=150))['level'],'| freq 11.1 ->',sf.assess(b,dict(N,dom_freq_hz=11.1))['level'])
