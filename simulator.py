"""
simulator.py  --  RL Gatekeeper Radar Inspection Dashboard
==========================================================
End-to-end Streamlit simulator for the hierarchical inspection pipeline:

  SIGNAL GENERATOR -> PREPROCESSING -> RL GATEKEEPER -> [ML CLASSIFIER] -> OUTPUT

The RL gatekeeper consumes the shared feature tensor X and decides DISCARD vs
FORWARD *before* the ML classifier runs.  The ML model is only invoked on
FORWARD decisions, so DISCARDs save compute -- the dashboard measures and
visualises that saving in real time.

Run:
    streamlit run simulator.py
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from src.rl.data_source import RadarSignalSource
from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile, _normalize_env
from src.data.preprocessing import compute_spectrogram
from src.models.cnn_lstm import build_model
from src.rl.dqn_agent import GatekeeperAgent
from src.rl.reward import RewardConfig, RewardFunction, derive_class_values
from src.rl.encoder import count_parameters
from src.explain import gradcam_on_rd_map
from src.rl.explain import gatekeeper_attribution, action_rationale

# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']
CLASS_ICONS = {'Drone': '🚁', 'Aircraft': '✈️', 'Bird': '🐦', 'Clutter': '🌧️', 'Noise': '📡'}
CLASS_COLORS = {'Drone': '#FF4136', 'Aircraft': '#FF851B', 'Bird': '#2ECC40',
                'Clutter': '#0074D9', 'Noise': '#888888'}

CFG = yaml.safe_load(open(Path(__file__).parent / 'configs' / 'config.yaml'))
THREAT_CLASSES = set(CFG['reward']['threat_classes'])
ML_PATH = Path(CFG['rl']['ml_model_path'])
GK_PATH = Path(CFG['rl']['save_path'])

DARK_BG, PAPER_BG, GRID_COL, TEXT_COL = '#0a0e1a', '#0d1525', '#1e3a5f', '#7aa3cc'

st.set_page_config(page_title="RL Gatekeeper Radar Inspector", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e0e6f0;}
[data-testid="stSidebar"]{background:#0d1525;border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#c8d8f0 !important;}
.pipeline-wrap{display:flex;align-items:stretch;justify-content:center;gap:0;
  padding:18px 10px;background:#0d1525;border:1px solid #1e3a5f;border-radius:14px;margin:10px 0;}
.pipe-block{flex:1;min-width:130px;max-width:200px;background:#111d33;border:2px solid #1e3a5f;
  border-radius:10px;padding:14px 10px;text-align:center;transition:all .3s ease;}
.pipe-block.active{border-color:#00d4ff;background:#0d2240;box-shadow:0 0 22px rgba(0,212,255,.55);}
.pipe-block.done{border-color:#2ECC40;background:#0d2215;}
.pipe-block.skip{border-color:#3a3a3a;background:#15151a;opacity:.45;}
.pipe-block.threat{border-color:#FF4136;background:#2a0d0d;}
.pipe-arrow{display:flex;align-items:center;padding:0 6px;color:#1e3a5f;font-size:1.4em;}
.pipe-arrow.active{color:#00d4ff;}
.block-icon{font-size:1.8em;margin-bottom:5px;}
.block-title{font-size:.7em;font-weight:700;color:#00d4ff;letter-spacing:1px;text-transform:uppercase;}
.block-sub{font-size:.6em;color:#5588aa;margin:4px 0;line-height:1.4;}
.block-data{font-size:.64em;color:#2ECC40;margin-top:6px;background:#0a1a0a;border-radius:4px;
  padding:3px 6px;min-height:18px;word-break:break-all;}
.block-data.red{color:#FF4136;background:#1a0a0a;}
.block-data.grey{color:#888;background:#15151a;}
.stat-card{background:#111d33;border:1px solid #1e3a5f;border-radius:10px;padding:14px;text-align:center;}
.stat-label{font-size:.66em;color:#7aa3cc;text-transform:uppercase;letter-spacing:1px;}
.stat-value{font-size:1.8em;font-weight:700;color:#00d4ff;margin-top:4px;}
.sim-header{background:linear-gradient(135deg,#0d1525,#1a2f50,#0d1525);border:1px solid #1e3a5f;
  border-radius:12px;padding:18px 30px;text-align:center;margin-bottom:16px;}
.sim-header h1{color:#00d4ff;font-size:1.9em;font-weight:700;letter-spacing:2px;margin:0;
  text-shadow:0 0 20px rgba(0,212,255,.5);}
.sim-header p{color:#7aa3cc;margin:4px 0 0;font-size:.85em;letter-spacing:1px;}
@keyframes pulse-active{0%{box-shadow:0 0 10px rgba(0,212,255,.3);}50%{box-shadow:0 0 28px rgba(0,212,255,.7);}
  100%{box-shadow:0 0 10px rgba(0,212,255,.3);}}
.pipe-block.active{animation:pulse-active 1.2s infinite;}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Resource loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml():
    if not ML_PATH.exists():
        return None, None
    ckpt = torch.load(ML_PATH, map_location='cpu', weights_only=False)
    model = build_model(ckpt.get('config', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


@st.cache_resource
def load_gatekeeper():
    if not GK_PATH.exists():
        return None, None
    agent = GatekeeperAgent.from_checkpoint(str(GK_PATH), device='cpu')
    ckpt = torch.load(GK_PATH, map_location='cpu', weights_only=False)
    return agent, ckpt


@st.cache_resource
def get_source():
    """Real 77 GHz Zenodo signal pool (matched to the 77 GHz ML model)."""
    rd_fft = CFG['preprocessing'].get('fft_size', 128)
    dop_fft = CFG['features'].get('doppler_sequence_length', 32)
    return RadarSignalSource(mode='zenodo', range_fft=rd_fft, doppler_fft=dop_fft,
                             seed=None, verbose=False, keep_iq=True)


@st.cache_resource
def get_reward_fn(_gk_ckpt):
    """Reconstruct the reward function from the gatekeeper checkpoint metadata."""
    rcfg = RewardConfig.from_dict(CFG.get('reward', {}))
    if _gk_ckpt and 'class_values' in _gk_ckpt:
        cv = np.asarray(_gk_ckpt['class_values'], dtype=np.float32)
        budget = _gk_ckpt.get('forward_budget', None)
    else:
        cv = derive_class_values(np.ones(5), threat_indices=[0, 1],
                                 threat_value=rcfg.threat_value,
                                 nonthreat_value=rcfg.nonthreat_value)
        budget = None
    return RewardFunction(rcfg, cv, forward_budget=budget)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline diagram
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_BLOCKS = [
    dict(id=0, icon='📡', title='77 GHz<br>SOURCE', sub='Real Zenodo FMCW segment'),
    dict(id=1, icon='⚡', title='PRE-<br>PROCESSING', sub='FFT → RD map [32×128]'),
    dict(id=2, icon='🤖', title='RL<br>GATEKEEPER', sub='Dueling DQN on X · ~33K p'),
    dict(id=3, icon='🧠', title='ML CLASSIFIER<br>77 GHz', sub='CNN-LSTM · only if FORWARD'),
    dict(id=4, icon='🎯', title='DECISION<br>OUTPUT', sub='Threat / Clear / Saved'),
]
ARROW_LABELS = ['IQ [32×128]', 'Tensor X', 'action', 'class p[5]']


def render_pipeline(active=-1, bd=None, ml_skipped=False, threat=False):
    bd = bd or {}
    done = active >= 5
    html = ''
    for b in PIPELINE_BLOCKS:
        bid = b['id']
        if bid == 3 and ml_skipped and (done or active > 3):
            state = 'skip'
        elif done and bid == 4 and threat:
            state = 'threat'
        elif done:
            state = 'done'
        elif bid == active:
            state = 'active'
        elif bid < active:
            state = 'done'
        else:
            state = 'idle'
        data = bd.get(bid, '—')
        dcls = 'red' if (bid == 4 and threat and done) else ('grey' if state == 'skip' else '')
        html += f"""<div class="pipe-block {state}">
          <div class="block-icon">{b['icon']}</div>
          <div class="block-title">{b['title']}</div>
          <div class="block-sub">{b['sub']}</div>
          <div class="block-data {dcls}">{data}</div></div>"""
        if bid < 4:
            aact = 'active' if (bid < active or done) else ''
            html += (f'<div class="pipe-arrow {aact}"><div style="text-align:center;">→'
                     f'<div style="font-size:.5em;color:#3a5a7a;">{ARROW_LABELS[bid]}</div></div></div>')
    return f'<div class="pipeline-wrap">{html}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def _base(h=260, **kw):
    return dict(paper_bgcolor=PAPER_BG, plot_bgcolor=DARK_BG,
                font=dict(color=TEXT_COL, family='monospace'),
                margin=dict(l=40, r=20, t=40, b=40), height=h, **kw)


def plot_rd(rd, title='Range-Doppler Map'):
    fig = go.Figure(go.Heatmap(z=rd, colorscale='Plasma',
                               colorbar=dict(tickfont=dict(color=TEXT_COL))))
    fig.update_layout(**_base(270, title=dict(text=title, font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(title='Range Bin', color=TEXT_COL, gridcolor=GRID_COL),
                      yaxis=dict(title='Doppler Bin', color=TEXT_COL, gridcolor=GRID_COL))
    return fig


def plot_profiles(rd):
    dop = rd.mean(axis=1)
    rng = rd.mean(axis=0)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Doppler Profile', 'Range Profile'))
    fig.add_trace(go.Scatter(y=dop, mode='lines', line=dict(color='#00d4ff')), 1, 1)
    fig.add_trace(go.Scatter(y=rng, mode='lines', line=dict(color='#2ECC40')), 1, 2)
    fig.update_layout(**_base(220, showlegend=False))
    fig.update_xaxes(gridcolor=GRID_COL, color=TEXT_COL)
    fig.update_yaxes(gridcolor=GRID_COL, color=TEXT_COL)
    for ann in fig.layout.annotations:
        ann.font.color = '#00d4ff'
    return fig


def plot_qvals(q, action):
    colors = ['#2ECC40' if i == action else '#1e3a5f' for i in range(2)]
    fig = go.Figure(go.Bar(x=['DISCARD', 'FORWARD'], y=q, marker_color=colors,
                           text=[f'{v:.3f}' for v in q], textposition='outside',
                           textfont=dict(color=TEXT_COL)))
    fig.update_layout(**_base(240, showlegend=False,
                      title=dict(text='Gatekeeper Q-values (on X)', font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(color=TEXT_COL), yaxis=dict(color=TEXT_COL, gridcolor=GRID_COL))
    return fig


def plot_probs(probs, highlight, title='ML Class Probabilities'):
    colors = [CLASS_COLORS[c] if c == highlight else '#1e3a5f' for c in CLASS_NAMES]
    fig = go.Figure(go.Bar(x=[f'{CLASS_ICONS[c]} {c}' for c in CLASS_NAMES], y=probs * 100,
                           marker_color=colors, text=[f'{p*100:.1f}%' for p in probs],
                           textposition='outside', textfont=dict(color=TEXT_COL)))
    fig.update_layout(**_base(240, showlegend=False,
                      title=dict(text=title, font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(color=TEXT_COL),
                      yaxis=dict(range=[0, 115], color=TEXT_COL, gridcolor=GRID_COL, title='Confidence (%)'))
    return fig


def plot_waveform(iq):
    pulse = iq[iq.shape[0] // 2]          # middle pulse (fast-time)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pulse.real, mode='lines', name='I', line=dict(color='#00d4ff')))
    fig.add_trace(go.Scatter(y=pulse.imag, mode='lines', name='Q', line=dict(color='#FF851B')))
    fig.update_layout(**_base(220, title=dict(text='Raw Waveform (I/Q, fast-time)',
                      font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(title='Sample', color=TEXT_COL, gridcolor=GRID_COL),
                      yaxis=dict(title='Amplitude', color=TEXT_COL, gridcolor=GRID_COL),
                      legend=dict(font=dict(color=TEXT_COL)))
    return fig


def plot_iq_constellation(iq):
    flat = iq.reshape(-1)
    fig = go.Figure(go.Scatter(x=flat.real, y=flat.imag, mode='markers',
                               marker=dict(size=3, color='#2ECC40', opacity=0.4)))
    fig.update_layout(**_base(220, title=dict(text='IQ Constellation', font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(title='In-phase (I)', color=TEXT_COL, gridcolor=GRID_COL, zeroline=True, zerolinecolor=GRID_COL),
                      yaxis=dict(title='Quadrature (Q)', color=TEXT_COL, gridcolor=GRID_COL, zeroline=True, zerolinecolor=GRID_COL))
    return fig


def plot_spectrogram_view(iq):
    avg = iq.mean(axis=0)                  # averaged pulse
    _, _, sxx = compute_spectrogram(avg, fs=10000, nperseg=32, nfft=128)
    fig = go.Figure(go.Heatmap(z=sxx, colorscale='Viridis',
                               colorbar=dict(tickfont=dict(color=TEXT_COL))))
    fig.update_layout(**_base(220, title=dict(text='STFT Spectrogram', font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(title='Time bin', color=TEXT_COL),
                      yaxis=dict(title='Freq bin', color=TEXT_COL))
    return fig


def plot_gradcam(rd, cam):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('RD Map', 'Grad-CAM (ML attention)'))
    fig.add_trace(go.Heatmap(z=rd, colorscale='Plasma', showscale=False), 1, 1)
    fig.add_trace(go.Heatmap(z=cam, colorscale='Inferno', showscale=False), 1, 2)
    fig.update_layout(**_base(260))
    fig.update_xaxes(color=TEXT_COL); fig.update_yaxes(color=TEXT_COL)
    for ann in fig.layout.annotations:
        ann.font.color = '#00d4ff'
    return fig


def plot_confusion(history):
    matrix = np.zeros((5, 5), dtype=int)
    for r in history:
        if r['forwarded']:
            matrix[CLASS_NAMES.index(r['true_class'])][CLASS_NAMES.index(r['ml_pred'])] += 1
    rs = matrix.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    norm = matrix / rs
    fig = go.Figure(go.Heatmap(z=norm, x=CLASS_NAMES, y=CLASS_NAMES, colorscale='Blues',
                               text=[[f'{norm[i][j]:.2f}' for j in range(5)] for i in range(5)],
                               texttemplate='%{text}', showscale=True,
                               colorbar=dict(tickfont=dict(color=TEXT_COL))))
    fig.update_layout(**_base(300, title=dict(text='ML Confusion (forwarded only, row-norm)',
                      font=dict(color='#00d4ff', size=13))),
                      xaxis=dict(title='Predicted', color=TEXT_COL),
                      yaxis=dict(title='True', color=TEXT_COL, autorange='reversed'))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution  (gatekeeper-first: ML only on FORWARD)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(signal, ml_model, agent, reward_fn, phs, animate=True):
    speed = st.session_state.get('sim_speed', 0.3)
    bd = {}
    true_class = CLASS_NAMES[signal.label]

    def upd(active, skip=False, threat=False):
        phs['pipeline'].markdown(render_pipeline(active, bd, ml_skipped=skip, threat=threat),
                                 unsafe_allow_html=True)
        if animate:
            time.sleep(speed)

    # Block 0 — real 77 GHz source (a measured Zenodo CPI segment)
    upd(0)
    iq = signal.iq
    bd[0] = f'{true_class} · IQ {iq.shape[0]}×{iq.shape[1]}'

    # Block 1 — preprocess  (the shared tensor X; identical path for ML + RL)
    upd(1)
    rd = signal.rd_map
    dop = signal.doppler
    env = signal.env
    bd[1] = f'X · RD {rd.shape[0]}×{rd.shape[1]}'
    phs['rd'].plotly_chart(plot_rd(rd, f'Range-Doppler — {true_class}'),
                           use_container_width=True, config={'displayModeBar': False},
                           key=f'rd{time.time()}')
    phs['prof'].plotly_chart(plot_profiles(rd), use_container_width=True,
                             config={'displayModeBar': False}, key=f'pf{time.time()}')

    # Block 2 — RL gatekeeper on X  (BEFORE any ML)
    upd(2)
    state = (rd, dop, env)
    t_gk = time.perf_counter()
    if agent is not None:
        q = agent.q_values(state)
        action = int(np.argmax(q))
    else:
        q = np.zeros(2, dtype=np.float32)
        action = 1  # no gatekeeper -> forward everything (baseline)
    gk_ms = (time.perf_counter() - t_gk) * 1e3
    forwarded = (action == 1)
    bd[2] = 'FORWARD' if forwarded else 'DISCARD'
    phs['rl'].plotly_chart(plot_qvals(q, action), use_container_width=True,
                           config={'displayModeBar': False}, key=f'q{time.time()}')

    # Block 3 — ML classifier  (only if forwarded)
    ml_probs = None; ml_pred = None; ml_conf = 0.0; ml_entropy = 0.0; ml_ms = 0.0
    if forwarded and ml_model is not None:
        upd(3, skip=False)
        t_ml = time.perf_counter()
        spec = torch.from_numpy(rd).unsqueeze(0).unsqueeze(0)
        d = torch.from_numpy(dop).unsqueeze(0); e = torch.from_numpy(env).unsqueeze(0)
        with torch.no_grad():
            ml_probs = F.softmax(ml_model(spec, d, e), dim=1).squeeze(0).numpy()
        ml_ms = (time.perf_counter() - t_ml) * 1e3
        ml_pred = CLASS_NAMES[int(ml_probs.argmax())]
        ml_conf = float(ml_probs.max())
        pe = np.clip(ml_probs, 1e-12, 1)
        ml_entropy = float(-np.sum(pe * np.log(pe)) / np.log(len(pe)))
        bd[3] = f'{ml_pred} · {ml_conf:.0%}'
        phs['ml'].plotly_chart(plot_probs(ml_probs, ml_pred), use_container_width=True,
                               config={'displayModeBar': False}, key=f'ml{time.time()}')
    else:
        bd[3] = 'ML SKIPPED ✓ saved'
        phs['ml'].markdown(
            '<div class="stat-card" style="padding:40px;color:#2ECC40;">'
            '🛑 ML inference skipped<br><span style="font-size:.8em;color:#7aa3cc;">'
            'gatekeeper discarded — compute saved</span></div>', unsafe_allow_html=True)

    # Block 4 — output
    is_true_threat = true_class in THREAT_CLASSES
    is_ml_threat = (ml_pred in THREAT_CLASSES) if ml_pred else False
    final_threat = forwarded and is_ml_threat
    if forwarded:
        bd[4] = ('⚠ THREAT' if is_ml_threat else '🔍 inspected')
    else:
        bd[4] = ('🚨 MISSED' if is_true_threat else '✓ cleared')
    upd(5, skip=not forwarded, threat=final_threat)

    # reward (real VoI reward, using ML probs when available; teacher probs otherwise)
    if ml_probs is None and ml_model is not None:
        spec = torch.from_numpy(rd).unsqueeze(0).unsqueeze(0)
        d = torch.from_numpy(dop).unsqueeze(0); e = torch.from_numpy(env).unsqueeze(0)
        with torch.no_grad():
            probs_for_reward = F.softmax(ml_model(spec, d, e), dim=1).squeeze(0).numpy()
    else:
        probs_for_reward = ml_probs if ml_probs is not None else np.ones(5) / 5
    reward, comp = reward_fn.reward(action, probs_for_reward, CLASS_NAMES.index(true_class))

    return {
        'true_class': true_class, 'forwarded': forwarded, 'action': bd[2],
        'q_vals': q, 'gk_ms': gk_ms, 'ml_ms': ml_ms,
        'ml_pred': ml_pred or '—', 'ml_conf': ml_conf, 'ml_entropy': ml_entropy,
        'ml_probs': ml_probs, 'rd': rd, 'dop': dop, 'env': env, 'iq': iq,
        'reward': reward, 'u_ml': comp['u_ml'],
        'ml_correct': int(ml_pred == true_class) if ml_pred else 0,
        'is_true_threat': is_true_threat,
        'missed_threat': (not forwarded) and is_true_threat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.setdefault('sim_speed', 0.3)
    st.session_state.setdefault('last', None)


init_state()


def update_stats(res):
    st.session_state.history.append(res)
    st.session_state.last = res


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
ml_model, ml_ckpt = load_ml()
agent, gk_ckpt = load_gatekeeper()
reward_fn = get_reward_fn(gk_ckpt)
source = get_source()
present_classes = [CLASS_NAMES[i] for i in source.present_labels]

st.markdown("""<div class="sim-header"><h1>🎯 RL GATEKEEPER RADAR INSPECTOR</h1>
<p>Real 77 GHz Source → Preprocessing → RL Gatekeeper (on X) → ML Classifier (only if forwarded) → Decision</p></div>""",
            unsafe_allow_html=True)

# Status chips
cols = st.columns(4)
gk_params = count_parameters(agent.policy_net) if agent else 0
ml_params = count_parameters(ml_model) if ml_model else 0
status = [
    ('77 GHz ML Teacher', ml_model is not None, f'{ml_params:,} params' if ml_model else 'run finetune_zenodo.py'),
    ('RL Gatekeeper', agent is not None, f'{gk_params:,} params' if agent else 'run train_rl.py'),
    ('Real 77 GHz source', len(source) > 0, f'{len(source):,} segments · {len(present_classes)} cls'),
    ('Compute ratio', bool(agent and ml_model), f'{ml_params/max(gk_params,1):.0f}× smaller gate' if (agent and ml_model) else '—'),
]
for col, (label, ok, sub) in zip(cols, status):
    c = '#2ECC40' if ok else '#FF4136'
    col.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div>'
                 f'<div style="color:{c};font-weight:700;margin:4px 0;">{"ONLINE" if ok else "OFFLINE"}</div>'
                 f'<div style="font-size:.62em;color:#3a5a7a;">{sub}</div></div>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('### ⚙️ Controls')
    target = st.selectbox('Target Class (real 77 GHz)',
                          ['🎲 Random'] + [f'{CLASS_ICONS[c]} {c}' for c in present_classes])
    target_class = None if 'Random' in target else target.split(' ', 1)[1]
    st.caption('Classes present in the real 77 GHz set: ' + ', '.join(present_classes))
    st.session_state.sim_speed = st.slider('Animation speed (s/block)', 0.0, 1.0, 0.3, 0.05)
    batch_n = st.select_slider('Batch size', [1, 5, 10, 25, 50, 100], value=10)
    step_btn = st.button('⚡ Step (1 signal)', use_container_width=True)
    batch_btn = st.button(f'🚀 Run batch ({batch_n})', use_container_width=True)
    explain_on = st.checkbox('🔍 Explainability (Grad-CAM + attribution)', value=True)
    if st.button('🗑️ Reset session', use_container_width=True):
        st.session_state.history = []
        st.session_state.last = None
        st.rerun()
    st.caption(f"Signals processed: {len(st.session_state.history)}")

# Pipeline + signal panels
st.markdown('##### 📟 Inspection pipeline')
pipeline_ph = st.empty()
hist = st.session_state.history
last = st.session_state.last
pipeline_ph.markdown(render_pipeline(5 if last else -1, bd=None,
                     ml_skipped=(last and not last['forwarded']),
                     threat=(last and last['forwarded'] and last['ml_pred'] in THREAT_CLASSES)) if last
                     else render_pipeline(-1), unsafe_allow_html=True)

c_rd, c_prof = st.columns(2)
rd_ph, prof_ph = c_rd.empty(), c_prof.empty()
c_rl, c_ml = st.columns(2)
rl_ph, ml_ph = c_rl.empty(), c_ml.empty()
phs = {'pipeline': pipeline_ph, 'rd': rd_ph, 'prof': prof_ph, 'rl': rl_ph, 'ml': ml_ph}


def pick():
    """Draw a real 77 GHz segment (specific class if selected, else balanced random)."""
    if target_class:
        return source.sample_label(CLASS_NAMES.index(target_class))
    return source.sample(balanced=True)


if step_btn:
    res = run_pipeline(pick(), ml_model, agent, reward_fn, phs, animate=True)
    update_stats(res)
elif batch_btn:
    prog = st.progress(0.0)
    for i in range(batch_n):
        res = run_pipeline(pick(), ml_model, agent, reward_fn, phs, animate=(i == batch_n - 1))
        update_stats(res)
        prog.progress((i + 1) / batch_n)
    prog.empty()

# ─────────────────────────────────────────────────────────────────────────────
# System / savings metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('##### 📊 System metrics & ML workload reduction')
hist = st.session_state.history
if hist:
    total = len(hist)
    fwd = sum(r['forwarded'] for r in hist)
    disc = total - fwd
    ml_ms = sum(r['ml_ms'] for r in hist)
    gk_ms = sum(r['gk_ms'] for r in hist)
    # latency if every signal had been forwarded (baseline)
    avg_ml = (ml_ms / fwd) if fwd else 0.0
    baseline_ms = gk_ms + avg_ml * total if avg_ml else ml_ms
    actual_ms = gk_ms + ml_ms
    saved = (1 - actual_ms / baseline_ms) * 100 if baseline_ms > 0 else 0.0
    missed = sum(r['missed_threat'] for r in hist)
    threats = sum(r['is_true_threat'] for r in hist)
    recall = (threats - missed) / threats if threats else 1.0
    fwd_acc = (sum(r['ml_correct'] for r in hist if r['forwarded']) / fwd) if fwd else 0.0
    cum_reward = sum(r['reward'] for r in hist)

    cols = st.columns(6)
    items = [('Total', total, '#00d4ff'), ('Forwarded', fwd, '#FF851B'),
             ('Discarded', disc, '#2ECC40'),
             ('Workload ↓', f'{disc/total*100:.0f}%', '#2ECC40'),
             ('Latency ↓', f'{saved:.0f}%', '#00d4ff'),
             ('Threat recall', f'{recall*100:.0f}%', '#2ECC40' if recall > 0.9 else '#FF4136')]
    for col, (l, v, c) in zip(cols, items):
        col.markdown(f'<div class="stat-card"><div class="stat-label">{l}</div>'
                     f'<div class="stat-value" style="color:{c};">{v}</div></div>', unsafe_allow_html=True)

    throughput = total / ((actual_ms / 1000) or 1e-9)   # signals / sec (gate+ML)
    cols2 = st.columns(6)
    items2 = [('Missed threats', missed, '#FF4136' if missed else '#2ECC40'),
              ('ML acc (fwd)', f'{fwd_acc*100:.0f}%', '#00d4ff'),
              ('Cumul. reward', f'{cum_reward:+.1f}', '#2ECC40' if cum_reward >= 0 else '#FF4136'),
              ('ML calls saved', f'{disc}', '#2ECC40'),
              ('Avg ML latency', f'{avg_ml:.1f} ms', '#00d4ff'),
              ('Throughput', f'{throughput:.0f}/s', '#00d4ff')]
    for col, (l, v, c) in zip(cols2, items2):
        col.markdown(f'<div class="stat-card" style="margin-top:6px;"><div class="stat-label">{l}</div>'
                     f'<div class="stat-value" style="font-size:1.4em;color:{c};">{v}</div></div>',
                     unsafe_allow_html=True)
else:
    st.info('Run a signal or batch to populate metrics.')

# Signal views (raw waveform, IQ samples, spectrogram, range/doppler, RD map)
if last is not None and last.get('iq') is not None:
    with st.expander('📡 Signal views (raw waveform · IQ samples · spectrogram · profiles · RD map)', expanded=False):
        iq = last['iq']; rd = last['rd']
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.plotly_chart(plot_waveform(iq), use_container_width=True,
                          config={'displayModeBar': False}, key='wv')
        r1c2.plotly_chart(plot_iq_constellation(iq), use_container_width=True,
                          config={'displayModeBar': False}, key='iq')
        r1c3.plotly_chart(plot_spectrogram_view(iq), use_container_width=True,
                          config={'displayModeBar': False}, key='spec')
        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(plot_rd(rd, f"Range-Doppler — {last['true_class']}"),
                          use_container_width=True, config={'displayModeBar': False}, key='rdv')
        r2c2.plotly_chart(plot_profiles(rd), use_container_width=True,
                          config={'displayModeBar': False}, key='pfv')

# Confusion matrix (forwarded signals)
if sum(r['forwarded'] for r in hist) >= 2:
    st.markdown('##### 🧠 ML monitoring')
    st.plotly_chart(plot_confusion(hist), use_container_width=True,
                    config={'displayModeBar': False}, key='conf')

# ─────────────────────────────────────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────────────────────────────────────
if explain_on and last is not None and ml_model is not None and agent is not None:
    st.markdown('##### 🔍 Explainability — why this decision?')
    rd, dop, env = last['rd'], last['dop'], last['env']

    attr = gatekeeper_attribution(agent.policy_net, rd, dop, env)
    st.markdown(f"**Gatekeeper:** {action_rationale(attr)}")
    bi = attr['branch_importance']
    bcols = st.columns(3)
    for col, k in zip(bcols, ['rd_map', 'doppler', 'env']):
        col.markdown(f'<div class="stat-card"><div class="stat-label">{k} attribution</div>'
                     f'<div class="stat-value" style="font-size:1.3em;">{bi[k]*100:.0f}%</div></div>',
                     unsafe_allow_html=True)

    if last['forwarded']:
        cam, ci, _ = gradcam_on_rd_map(ml_model, rd, dop, env)
        st.plotly_chart(plot_gradcam(rd, cam), use_container_width=True,
                        config={'displayModeBar': False}, key='gcam')
        st.caption(f"Grad-CAM shows which Range-Doppler regions drove the ML "
                   f"classification as **{last['ml_pred']}**.")
    else:
        st.caption("Signal was discarded — ML Grad-CAM not applicable (no ML inference ran).")

st.markdown("""<hr style="border:none;border-top:1px solid #1e3a5f;margin:24px 0 10px;">
<div style="text-align:center;color:#3a5a7a;font-size:.72em;">
RL Gatekeeper · Value-of-Information reward · Dueling DQN + PER · CNN-LSTM 77 GHz teacher · IDP 2026</div>""",
            unsafe_allow_html=True)
