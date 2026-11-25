"""IntelliBridge - Streamlit version of the live predictor (uses the real Python model)."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import pandas as pd, streamlit as st
import intellibridge_predict as ib, intellibridge_safety as sf

st.set_page_config(page_title='IntelliBridge', page_icon='🌉', layout='wide')
TEAM = ['Yash Raj Singh', 'Manish Patel', 'N. Swathi', 'Aditya Jayant', 'Yashodhara Singh', 'Likitha Vijayakumar']
PRESETS = {'Normal day': [3, 4, .05, 0, 20, 0, 20, 65], 'Heavy truck, hot day': [9.5, 3.9, .05, .3, 20, 130, 38, 40],
           'Early damage': [3.1, 3.5, .4, .5, 15, 90, 22, 65], 'Flood': [3, 4, .05, 0, 210, 0, 20, 90],
           'Severe damage': [4, 3.1, 1.4, 1.8, 20, 220, 20, 65]}
COLOR = {'Safe': '#169b6b', 'Warning': '#d98a00', 'Danger': '#d93a41'}

@st.cache_resource(show_spinner='Training the model (about 20 seconds, first visit only)...')
def get_model():
    return sf.load(os.path.join(tempfile.gettempdir(), 'intellibridge_model.joblib'))

def set_preset(name):
    for k, v in zip(ib.FEATS, PRESETS[name]): st.session_state[k] = float(v)

for k in ib.FEATS: st.session_state.setdefault(k, float(ib.TYPICAL[k]))
st.title('IntelliBridge')
st.caption('A low-cost, offline multi-sensor framework for real-time bridge health risk classification')
st.write(' | '.join(TEAM))
st.write('Type any sensor readings. The model returns a danger level from 1 to 100.')
for c, n in zip(st.columns(len(PRESETS)), PRESETS): c.button(n, on_click=set_preset, args=(n,), use_container_width=True)

left, right = st.columns([1.1, .9], gap='large')
with left:
    reading = {k: st.number_input(f'{ib.PROMPTS[k][0]} ({ib.PROMPTS[k][1]})', key=k, step=0.1, format='%g', help=ib.PROMPTS[k][2]) for k in ib.FEATS}
with right:
    a = sf.assess(get_model(), reading)
    if a['status'] != 'ok':
        st.error('No prediction: ' + '; '.join(a['errors']))
    else:
        col = COLOR[a['cls']]
        st.markdown(f"<div style='font-size:64px;font-weight:800;color:{col};line-height:1'>{a['level']}<span style='font-size:22px'> / 100</span></div>"
                    f"<div style='font-size:26px;font-weight:700;color:{col}'>{a['cls'].upper()}</div>", unsafe_allow_html=True)
        st.progress(a['level'] / 100)
        st.caption('1-29 Safe | 30-64 Warning | 65-100 Danger')
        if a['why']: st.write('**Main reasons:** ' + ', '.join(a['why']))
        st.info('**Action:** ' + a['action'])
        for n in a['notes']: st.warning(n)
        st.write('A fixed-threshold rule would say: **' + str(ib.fixed_threshold_rule(pd.DataFrame([reading]))[0]).upper() + '**')
st.caption('Model trained on simulated data. It shows the method, not real-world accuracy.')
