"""
Radar Threat Classification - Interactive UI

Dual-model analysis system:
  Model A : 24 GHz Synthetic (outputs/models/best_model.pt)
  Model B : 77 GHz Fine-tuned (outputs/models/best_model_zenodo.pt)

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
# import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from src.data.synthetic_generator import SyntheticRadarGenerator
from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile
from src.models.cnn_lstm import build_model

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES   = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']
THREAT_LEVELS = {'Drone': 0.90, 'Aircraft': 0.80, 'Bird': 0.20,
                 'Clutter': 0.10, 'Noise': 0.00}
CLASS_COLORS  = {
    'Drone':    '#FF4136',
    'Aircraft': '#FF851B',
    'Bird':     '#2ECC40',
    'Clutter':  '#0074D9',
    'Noise':    '#AAAAAA',
}
CLASS_ICONS   = {
    'Drone': '🚁', 'Aircraft': '✈️', 'Bird': '🐦',
    'Clutter': '🌧️', 'Noise': '📡',
}

MODEL_A_PATH = Path('outputs/models/best_model.pt')
MODEL_B_PATH = Path('outputs/models/best_model_zenodo.pt')

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Radar Threat Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark radar theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e0e6f0;
}
[data-testid="stSidebar"] {
    background: #0d1525;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* ── Header banner ── */
.radar-header {
    background: linear-gradient(135deg, #0d1525 0%, #1a2f50 50%, #0d1525 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 20px;
    text-align: center;
}
.radar-header h1 {
    color: #00d4ff;
    font-size: 2.2em;
    font-weight: 700;
    letter-spacing: 3px;
    margin: 0;
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
}
.radar-header p {
    color: #7aa3cc;
    margin: 5px 0 0 0;
    font-size: 0.95em;
    letter-spacing: 1px;
}

/* ── Metric cards ── */
.metric-card {
    background: #111d33;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin: 4px;
}
.metric-card .label {
    font-size: 0.75em;
    color: #7aa3cc;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card .value {
    font-size: 1.8em;
    font-weight: 700;
    color: #00d4ff;
    margin-top: 4px;
}

/* ── Threat badge ── */
.threat-high   { color: #FF4136; font-weight: bold; font-size: 1.3em; }
.threat-medium { color: #FF851B; font-weight: bold; font-size: 1.3em; }
.threat-low    { color: #2ECC40; font-weight: bold; font-size: 1.3em; }
.threat-none   { color: #AAAAAA; font-weight: bold; font-size: 1.3em; }

/* ── Section header ── */
.section-title {
    color: #00d4ff;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 16px 0 10px 0;
}

/* ── Prediction label ── */
.pred-box {
    background: #111d33;
    border-left: 4px solid #00d4ff;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.pred-class  { font-size: 1.5em; font-weight: 700; }
.pred-conf   { font-size: 0.85em; color: #7aa3cc; margin-top: 2px; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #7aa3cc !important;
    font-size: 0.9em;
    letter-spacing: 1px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a3a6e, #0d2444);
    color: #00d4ff;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 1px;
    width: 100%;
}
[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1e4a8e, #1a3a6e);
    border-color: #00d4ff;
}

/* ── Sliders / selects ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label { color: #7aa3cc !important; }

/* ── Model status pill ── */
.model-online  { display:inline-block; background:#0d2a1a; color:#2ECC40;
                 border:1px solid #2ECC40; border-radius:20px;
                 padding:2px 12px; font-size:0.75em; }
.model-offline { display:inline-block; background:#2a0d0d; color:#FF4136;
                 border:1px solid #FF4136; border-radius:20px;
                 padding:2px 12px; font-size:0.75em; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading  (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None, None
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt.get('config', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


@st.cache_resource
def get_generator():
    return SyntheticRadarGenerator(seed=None)


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, rd_map: np.ndarray):
    """Returns (class_idx, confidence, probs[5])"""
    spec = torch.from_numpy(rd_map).unsqueeze(0).unsqueeze(0)
    dop  = torch.from_numpy(rd_map.mean(axis=1)).unsqueeze(0)
    env  = torch.zeros(1, 3)
    logits = model(spec, dop, env)
    probs  = F.softmax(logits, dim=1).squeeze().numpy()
    idx    = int(probs.argmax())
    return idx, float(probs[idx]), probs


@torch.no_grad()
def predict_ensemble(model, cls_name, extra_params, n, fix_seed, seed_val):
    """
    Run inference on N independently generated samples and average probabilities.
    Averaging probs (not RD maps) avoids range-bin smearing and gives a stable,
    statistically robust prediction.
    """
    gen      = get_generator()
    prob_sum = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for i in range(n):
        if fix_seed:
            np.random.seed(seed_val + i)
        iq, _ = gen.generate_sample(cls_name, **extra_params)
        rd     = iq_to_range_doppler(iq)
        _, _, p = predict(model, rd)
        prob_sum += p
    avg_probs = prob_sum / n
    idx = int(avg_probs.argmax())
    return idx, float(avg_probs[idx]), avg_probs


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG  = '#0a0e1a'
PAPER_BG = '#0d1525'
GRID_COL = '#1e3a5f'
TEXT_COL = '#7aa3cc'

def _base_layout(**kwargs):
    return dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COL, family='monospace'),
        margin=dict(l=40, r=20, t=40, b=40),
        **kwargs,
    )


def plot_rd_map(rd_map: np.ndarray, title='Range-Doppler Map') -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=rd_map,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(
            title=dict(text='Power (norm)', font=dict(color=TEXT_COL)),
            tickfont=dict(color=TEXT_COL),
        ),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text=title, font=dict(color='#00d4ff', size=14))),
        xaxis=dict(title='Range Bin', color=TEXT_COL, gridcolor=GRID_COL,
                   showgrid=True, zeroline=False),
        yaxis=dict(title='Doppler Bin', color=TEXT_COL, gridcolor=GRID_COL,
                   showgrid=True, zeroline=False),
        height=280,
    )
    return fig


def plot_doppler_profile(rd_map: np.ndarray, pred_class: str = None) -> go.Figure:
    profile = rd_map.mean(axis=1)
    color   = CLASS_COLORS.get(pred_class, '#00d4ff')
    # Convert hex to rgba for fill (Plotly 6 doesn't support 8-digit hex)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    fill_color = f'rgba({r},{g},{b},0.15)'
    fig = go.Figure(go.Scatter(
        y=profile,
        mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy',
        fillcolor=fill_color,
    ))
    fig.update_layout(
        **_base_layout(title=dict(text='Doppler Profile', font=dict(color='#00d4ff', size=14))),
        xaxis=dict(title='Doppler Bin', color=TEXT_COL, gridcolor=GRID_COL,
                   showgrid=True, zeroline=False),
        yaxis=dict(title='Normalised Power', color=TEXT_COL, gridcolor=GRID_COL,
                   showgrid=True, zeroline=False),
        height=220,
    )
    return fig


def plot_confidence_bars(probs: np.ndarray, highlight_idx: int) -> go.Figure:
    colors = [
        CLASS_COLORS[cls] if i == highlight_idx else '#1e3a5f'
        for i, cls in enumerate(CLASS_NAMES)
    ]
    fig = go.Figure(go.Bar(
        x=[f"{CLASS_ICONS[c]} {c}" for c in CLASS_NAMES],
        y=probs * 100,
        marker_color=colors,
        text=[f'{p*100:.1f}%' for p in probs],
        textposition='outside',
        textfont=dict(color=TEXT_COL),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text='Class Probabilities (%)',
                                  font=dict(color='#00d4ff', size=14))),
        xaxis=dict(color=TEXT_COL, gridcolor=GRID_COL),
        yaxis=dict(range=[0, 115], color=TEXT_COL, gridcolor=GRID_COL,
                   showgrid=True, zeroline=False),
        height=260,
        showlegend=False,
    )
    return fig


def plot_threat_gauge(threat_level: float, class_name: str) -> go.Figure:
    if threat_level >= 0.7:
        color = '#FF4136'
    elif threat_level >= 0.4:
        color = '#FF851B'
    elif threat_level >= 0.1:
        color = '#FFDC00'
    else:
        color = '#2ECC40'

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=threat_level * 100,
        number=dict(suffix='%', font=dict(color=color, size=32)),
        title=dict(text=f'Threat Level<br><b>{class_name}</b>',
                   font=dict(color='#7aa3cc', size=13)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_COL,
                      tickfont=dict(color=TEXT_COL)),
            bar=dict(color=color, thickness=0.3),
            bgcolor=DARK_BG,
            bordercolor=GRID_COL,
            steps=[
                dict(range=[0, 20],  color='#0d2a1a'),
                dict(range=[20, 40], color='#1a2a0d'),
                dict(range=[40, 70], color='#2a1e0d'),
                dict(range=[70, 100], color='#2a0d0d'),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.75,
                value=threat_level * 100,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COL, family='monospace'),
        margin=dict(l=20, r=20, t=60, b=20),
        height=260,
    )
    return fig


def plot_comparison_bar(probs_a, probs_b) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='24GHz Synthetic',
        x=[f"{CLASS_ICONS[c]} {c}" for c in CLASS_NAMES],
        y=probs_a * 100,
        marker_color='#00d4ff',
        opacity=0.85,
        text=[f'{p*100:.1f}%' for p in probs_a],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='77GHz Fine-tuned',
        x=[f"{CLASS_ICONS[c]} {c}" for c in CLASS_NAMES],
        y=probs_b * 100,
        marker_color='#FF851B',
        opacity=0.85,
        text=[f'{p*100:.1f}%' for p in probs_b],
        textposition='outside',
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text='Model A vs Model B — Confidence Comparison',
                       font=dict(color='#00d4ff', size=14)),
            barmode='group',
        ),
        xaxis=dict(color=TEXT_COL, gridcolor=GRID_COL),
        yaxis=dict(range=[0, 120], color=TEXT_COL, gridcolor=GRID_COL,
                   title='Confidence (%)', showgrid=True, zeroline=False),
        legend=dict(bgcolor=PAPER_BG, bordercolor=GRID_COL,
                    font=dict(color=TEXT_COL)),
        height=320,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ──────────────────────────────────────────────────────────────────────────────

def threat_badge(level: float) -> str:
    if level >= 0.7:
        return f'<span class="threat-high">🔴 HIGH ({level:.0%})</span>'
    elif level >= 0.4:
        return f'<span class="threat-medium">🟠 MEDIUM ({level:.0%})</span>'
    elif level >= 0.1:
        return f'<span class="threat-low">🟡 LOW ({level:.0%})</span>'
    else:
        return f'<span class="threat-none">⚪ NONE ({level:.0%})</span>'


def result_card(class_name: str, confidence: float, model_label: str):
    color  = CLASS_COLORS[class_name]
    icon   = CLASS_ICONS[class_name]
    threat = THREAT_LEVELS[class_name]
    st.markdown(f"""
    <div class="pred-box" style="border-left-color:{color};">
        <div style="font-size:0.7em;color:#7aa3cc;letter-spacing:1px;">
            {model_label}
        </div>
        <div class="pred-class" style="color:{color};">
            {icon} {class_name}
        </div>
        <div class="pred-conf">
            Confidence: <b style="color:{color}">{confidence:.1%}</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Threat: {threat_badge(threat)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────

if 'rd_map'           not in st.session_state: st.session_state.rd_map           = None
if 'true_class'       not in st.session_state: st.session_state.true_class       = None
if 'history'          not in st.session_state: st.session_state.history          = []
if 'n_avg'            not in st.session_state: st.session_state.n_avg            = 1
if 'fix_seed'         not in st.session_state: st.session_state.fix_seed         = True
if 'seed_val'         not in st.session_state: st.session_state.seed_val         = 42
if 'extra_params'     not in st.session_state: st.session_state.extra_params     = {}
if 'target_class_gen' not in st.session_state: st.session_state.target_class_gen = None


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="radar-header">
    <h1>🎯 RADAR THREAT CLASSIFIER</h1>
    <p>Dual-Model Real-Time Analysis System &nbsp;|&nbsp;
       24 GHz Synthetic &amp; 77 GHz Fine-tuned</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-title">⚙️ System Status</div>',
                unsafe_allow_html=True)

    model_a, ckpt_a = load_model(MODEL_A_PATH)
    model_b, ckpt_b = load_model(MODEL_B_PATH)

    def _pill(ok, label):
        cls = 'model-online' if ok else 'model-offline'
        txt = 'ONLINE' if ok else 'OFFLINE'
        st.markdown(f'<b>{label}</b>&nbsp;&nbsp;<span class="{cls}">{txt}</span>',
                    unsafe_allow_html=True)

    _pill(model_a is not None, '24 GHz Synthetic')
    if ckpt_a:
        st.caption(f"Val acc: {ckpt_a.get('val_acc', 0):.1f}%")
    _pill(model_b is not None, '77 GHz Fine-tuned')
    if ckpt_b:
        st.caption(f"Val loss: {ckpt_b.get('val_loss', 0):.4f}")

    st.markdown('<div class="section-title">🎛️ Signal Generator</div>',
                unsafe_allow_html=True)

    target_class = st.selectbox(
        'Target Class',
        CLASS_NAMES,
        format_func=lambda c: f'{CLASS_ICONS[c]} {c}',
    )

    snr_db = st.slider('SNR (dB)', min_value=0, max_value=40, value=20)

    # Class-specific params
    extra_params = {}
    if target_class == 'Drone':
        v = st.slider('Velocity (m/s)', 0, 30, 10)
        r = st.slider('Rotor freq (Hz)', 50, 200, 100)
        extra_params = {'velocity': float(v), 'rotor_freq': float(r), 'snr_db': float(snr_db)}
    elif target_class == 'Aircraft':
        extra_params = {'snr_db': float(snr_db)}
    elif target_class == 'Bird':
        v  = st.slider('Velocity (m/s)', 5, 25, 12)
        wf = st.slider('Wing-beat (Hz)', 2, 15, 7)
        extra_params = {'velocity': float(v), 'wingbeat_freq': float(wf), 'snr_db': float(snr_db)}
    elif target_class == 'Clutter':
        ct = st.selectbox('Clutter type', ['ground', 'weather', 'sea'])
        extra_params = {'clutter_type': ct, 'snr_db': float(snr_db)}
    elif target_class == 'Noise':
        nt = st.selectbox('Noise type', ['thermal', 'phase', 'quantization'])
        extra_params = {'noise_type': nt}

    st.markdown('<div class="section-title">🎲 Reproducibility</div>',
                unsafe_allow_html=True)
    fix_seed   = st.checkbox('Fix random seed', value=True,
                             help='Same seed = same noise every click')
    seed_val   = st.slider('Seed', 0, 999, 42,
                           disabled=not fix_seed)
    n_avg      = st.select_slider(
        'Prediction averaging (N samples)',
        options=[1, 3, 5, 10, 20],
        value=5,
        help='Generate N samples with same params, average probabilities — reduces noise variance',
    )

    gen_btn = st.button('⚡ Generate Sample', use_container_width=True)

    st.markdown('<div class="section-title">📁 Upload IQ Data</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader('Upload .npy IQ matrix [32×128]',
                                type=['npy'], label_visibility='collapsed')

    if uploaded:
        try:
            iq = np.load(uploaded, allow_pickle=False)
            if iq.shape == (32, 128):
                st.session_state.rd_map     = iq_to_range_doppler(iq)
                st.session_state.true_class = 'Unknown'
                st.success('File loaded OK')
            else:
                st.error(f'Expected shape (32,128), got {iq.shape}')
        except Exception as e:
            st.error(f'Load error: {e}')

    st.markdown('<div class="section-title">📋 Prediction History</div>',
                unsafe_allow_html=True)
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            col = CLASS_COLORS[h['pred']]
            st.markdown(
                f'<span style="color:{col}">● {CLASS_ICONS[h["pred"]]} {h["pred"]}</span>'
                f'&nbsp;<span style="color:#7aa3cc;font-size:0.8em">'
                f'{h["conf"]:.0%} | true:{h["true"]}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption('No predictions yet.')

    if st.button('🗑️ Clear History'):
        st.session_state.history = []


# ──────────────────────────────────────────────────────────────────────────────
# Generate sample
# ──────────────────────────────────────────────────────────────────────────────

if gen_btn:
    gen = get_generator()
    with st.spinner(f'Generating {n_avg} sample(s) ...'):
        # Use fixed seed for the representative RD map shown in visualisation
        if fix_seed:
            np.random.seed(seed_val)
        iq_repr, _ = gen.generate_sample(target_class, **extra_params)
        st.session_state.rd_map          = iq_to_range_doppler(iq_repr)
        st.session_state.true_class      = target_class
        st.session_state.n_avg           = n_avg
        st.session_state.fix_seed        = fix_seed
        st.session_state.seed_val        = seed_val
        st.session_state.extra_params    = extra_params
        st.session_state.target_class_gen = target_class
    label = f'seed={seed_val}' if fix_seed else 'random seed'
    st.toast(
        f'{CLASS_ICONS[target_class]} {target_class} — {label}',
        icon='✅',
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main tabs
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    '📡  Single Analysis',
    '⚖️  Model Comparison',
    '📊  Live Dashboard',
])

rd_map = st.session_state.rd_map


# ── Tab 1: Single Analysis ─────────────────────────────────────────────────────
with tab1:
    if rd_map is None:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#7aa3cc;">
            <div style="font-size:3em;">📡</div>
            <div style="font-size:1.2em;margin-top:10px;">
                Generate a sample or upload an IQ file to begin analysis
            </div>
            <div style="font-size:0.85em;margin-top:8px;opacity:0.6;">
                Use the sidebar controls on the left
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Choose model
        model_choice = st.radio(
            'Active model',
            ['24 GHz Synthetic', '77 GHz Fine-tuned'],
            horizontal=True,
        )
        model = model_a if model_choice == '24 GHz Synthetic' else model_b
        model_label = model_choice

        if model is None:
            st.warning(f'{model_choice} model not found. Run train.py / finetune_zenodo.py first.')
        else:
            n        = st.session_state.n_avg
            use_ens  = (st.session_state.target_class_gen is not None and n > 1)
            with st.spinner(f'Running {"ensemble " if use_ens else ""}inference'
                            f'{f" (N={n})" if use_ens else ""} ...'):
                if use_ens:
                    idx, conf, probs = predict_ensemble(
                        model,
                        st.session_state.target_class_gen,
                        st.session_state.extra_params,
                        n,
                        st.session_state.fix_seed,
                        st.session_state.seed_val,
                    )
                else:
                    idx, conf, probs = predict(model, rd_map)
            pred_class = CLASS_NAMES[idx]
            threat_lvl = THREAT_LEVELS[pred_class]

            # Ensemble badge
            if use_ens:
                st.caption(f'Ensemble prediction averaged over {n} samples '
                           f'({"seed=" + str(st.session_state.seed_val) if st.session_state.fix_seed else "random seeds"})')

            # ── Top result card ─────────────────────────────────────────────
            result_card(pred_class, conf, model_label)

            # Append to history
            st.session_state.history.append({
                'pred': pred_class, 'conf': conf,
                'true': st.session_state.true_class or '?',
            })

            # ── Visualisations ───────────────────────────────────────────────
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.plotly_chart(
                    plot_rd_map(rd_map,
                                f'Range-Doppler Map  [{st.session_state.true_class}]'),
                    use_container_width=True, config={'displayModeBar': False},
                    key='t1_rd_map',
                )
                st.plotly_chart(
                    plot_doppler_profile(rd_map, pred_class),
                    use_container_width=True, config={'displayModeBar': False},
                    key='t1_doppler',
                )

            with col_right:
                st.plotly_chart(
                    plot_threat_gauge(threat_lvl, pred_class),
                    use_container_width=True, config={'displayModeBar': False},
                    key='t1_gauge',
                )
                st.plotly_chart(
                    plot_confidence_bars(probs, idx),
                    use_container_width=True, config={'displayModeBar': False},
                    key='t1_conf_bars',
                )

            # ── Stats row ────────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            for col, label, val in [
                (c1, 'Predicted Class',    f'{CLASS_ICONS[pred_class]} {pred_class}'),
                (c2, 'Confidence',         f'{conf:.1%}'),
                (c3, 'Threat Level',       f'{threat_lvl:.0%}'),
                (c4, 'True Label',         st.session_state.true_class or 'Unknown'),
            ]:
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{label}</div>'
                    f'<div class="value">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Correct / wrong indicator
            true_cl = st.session_state.true_class
            if true_cl and true_cl != 'Unknown':
                if pred_class == true_cl:
                    st.success(f'✅ Correct prediction — {model_label} identified {pred_class}')
                else:
                    st.error(f'❌ Misclassified — predicted {pred_class}, true: {true_cl}')


# ── Tab 2: Model Comparison ────────────────────────────────────────────────────
with tab2:
    if rd_map is None:
        st.info('Generate a sample first (sidebar).')
    elif model_a is None or model_b is None:
        missing = []
        if model_a is None: missing.append('24GHz model (run train.py)')
        if model_b is None: missing.append('77GHz model (run finetune_zenodo.py)')
        st.warning('Missing: ' + '  |  '.join(missing))
    else:
        with st.spinner('Running both models ...'):
            idx_a, conf_a, probs_a = predict(model_a, rd_map)
            idx_b, conf_b, probs_b = predict(model_b, rd_map)

        pred_a = CLASS_NAMES[idx_a]
        pred_b = CLASS_NAMES[idx_b]

        # ── RD map (shared input) ────────────────────────────────────────────
        st.plotly_chart(
            plot_rd_map(rd_map,
                        f'Shared Input — Range-Doppler Map  [{st.session_state.true_class}]'),
            use_container_width=True, config={'displayModeBar': False},
            key='t2_rd_map',
        )

        # ── Side-by-side results ─────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">MODEL A — 24 GHz Synthetic</div>',
                        unsafe_allow_html=True)
            result_card(pred_a, conf_a, '24 GHz Synthetic')
            st.plotly_chart(
                plot_confidence_bars(probs_a, idx_a),
                use_container_width=True, config={'displayModeBar': False},
                key='t2_conf_a',
            )
            st.plotly_chart(
                plot_threat_gauge(THREAT_LEVELS[pred_a], pred_a),
                use_container_width=True, config={'displayModeBar': False},
                key='t2_gauge_a',
            )

        with col_b:
            st.markdown('<div class="section-title">MODEL B — 77 GHz Fine-tuned</div>',
                        unsafe_allow_html=True)
            result_card(pred_b, conf_b, '77 GHz Fine-tuned')
            st.plotly_chart(
                plot_confidence_bars(probs_b, idx_b),
                use_container_width=True, config={'displayModeBar': False},
                key='t2_conf_b',
            )
            st.plotly_chart(
                plot_threat_gauge(THREAT_LEVELS[pred_b], pred_b),
                use_container_width=True, config={'displayModeBar': False},
                key='t2_gauge_b',
            )

        # ── Agreement / divergence ───────────────────────────────────────────
        st.plotly_chart(
            plot_comparison_bar(probs_a, probs_b),
            use_container_width=True, config={'displayModeBar': False},
            key='t2_comparison',
        )

        if pred_a == pred_b:
            st.success(f'✅ Both models agree: **{CLASS_ICONS[pred_a]} {pred_a}**')
        else:
            st.warning(
                f'⚠️ Models disagree — '
                f'Model A: **{pred_a}** ({conf_a:.1%}) | '
                f'Model B: **{pred_b}** ({conf_b:.1%})'
            )
            true_cl = st.session_state.true_class
            if true_cl and true_cl != 'Unknown':
                a_right = pred_a == true_cl
                b_right = pred_b == true_cl
                if a_right:
                    st.info(f'Model A is correct (true: {true_cl})')
                elif b_right:
                    st.info(f'Model B is correct (true: {true_cl})')


# ── Tab 3: Live Dashboard ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">⚡ RAPID FIRE — Generate & Classify</div>',
                unsafe_allow_html=True)

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        batch_n    = st.slider('Samples to generate', 5, 50, 20)
        batch_cls  = st.multiselect('Classes to include', CLASS_NAMES,
                                    default=CLASS_NAMES)
        dash_model = st.radio('Model', ['24 GHz Synthetic', '77 GHz Fine-tuned',
                                        'Both (Ensemble)'])
        run_dash   = st.button('▶ Run Batch', use_container_width=True)

    with col_results:
        if run_dash:
            gen = get_generator()

            mdl_a_active = dash_model in ('24 GHz Synthetic', 'Both (Ensemble)')
            mdl_b_active = dash_model in ('77 GHz Fine-tuned', 'Both (Ensemble)')
            use_a = mdl_a_active and model_a is not None
            use_b = mdl_b_active and model_b is not None

            if not (use_a or use_b):
                st.warning('No model available for selected option.')
            else:
                results = []
                prog = st.progress(0, text='Generating ...')

                for i in range(batch_n):
                    cls = np.random.choice(batch_cls)
                    iq, _ = gen.generate_sample(cls)
                    rd    = iq_to_range_doppler(iq)

                    if use_a and use_b:
                        _, _, pa = predict(model_a, rd)
                        _, _, pb = predict(model_b, rd)
                        probs  = (pa + pb) / 2
                    elif use_a:
                        _, _, probs = predict(model_a, rd)
                    else:
                        _, _, probs = predict(model_b, rd)

                    pidx  = int(probs.argmax())
                    pname = CLASS_NAMES[pidx]
                    results.append({
                        'true': cls, 'pred': pname,
                        'conf': float(probs[pidx]),
                        'correct': cls == pname,
                    })
                    prog.progress((i + 1) / batch_n,
                                  text=f'Sample {i+1}/{batch_n} ...')

                prog.empty()

                # Accuracy
                acc = np.mean([r['correct'] for r in results]) * 100
                c1, c2, c3 = st.columns(3)
                for col, label, val in [
                    (c1, 'Samples', str(batch_n)),
                    (c2, 'Accuracy', f'{acc:.1f}%'),
                    (c3, 'Avg Conf', f'{np.mean([r["conf"] for r in results]):.1%}'),
                ]:
                    col.markdown(
                        f'<div class="metric-card"><div class="label">{label}</div>'
                        f'<div class="value">{val}</div></div>',
                        unsafe_allow_html=True,
                    )

                # Per-class accuracy chart
                cls_acc = {}
                for cls in batch_cls:
                    cls_res = [r for r in results if r['true'] == cls]
                    if cls_res:
                        cls_acc[cls] = np.mean([r['correct'] for r in cls_res]) * 100

                if cls_acc:
                    fig = go.Figure(go.Bar(
                        x=[f'{CLASS_ICONS[c]} {c}' for c in cls_acc],
                        y=list(cls_acc.values()),
                        marker_color=[CLASS_COLORS[c] for c in cls_acc],
                        text=[f'{v:.0f}%' for v in cls_acc.values()],
                        textposition='outside',
                        textfont=dict(color=TEXT_COL),
                    ))
                    fig.update_layout(
                        **_base_layout(
                            title=dict(text='Per-Class Accuracy',
                                       font=dict(color='#00d4ff', size=14))
                        ),
                        yaxis=dict(range=[0, 115], color=TEXT_COL,
                                   gridcolor=GRID_COL, zeroline=False),
                        xaxis=dict(color=TEXT_COL),
                        height=280,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={'displayModeBar': False},
                                    key='t3_class_acc')

                # Prediction timeline
                timeline_fig = go.Figure()
                for cls in batch_cls:
                    xs = [i for i, r in enumerate(results) if r['true'] == cls]
                    ys = [results[i]['conf'] * 100 for i in xs]
                    correct = [results[i]['correct'] for i in xs]
                    timeline_fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode='markers',
                        name=f'{CLASS_ICONS[cls]} {cls}',
                        marker=dict(
                            color=CLASS_COLORS[cls],
                            size=[12 if c else 8 for c in correct],
                            symbol=['circle' if c else 'x' for c in correct],
                            line=dict(color='white', width=1),
                        ),
                    ))
                timeline_fig.update_layout(
                    **_base_layout(
                        title=dict(text='Prediction Timeline (circles=correct, x=wrong)',
                                   font=dict(color='#00d4ff', size=13))
                    ),
                    xaxis=dict(title='Sample index', color=TEXT_COL,
                               gridcolor=GRID_COL),
                    yaxis=dict(title='Confidence (%)', color=TEXT_COL,
                               gridcolor=GRID_COL, zeroline=False),
                    legend=dict(bgcolor=PAPER_BG, bordercolor=GRID_COL,
                                font=dict(color=TEXT_COL)),
                    height=280,
                )
                st.plotly_chart(timeline_fig, use_container_width=True,
                                config={'displayModeBar': False},
                                key='t3_timeline')
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#7aa3cc;">
                <div style="font-size:2em;">📊</div>
                <div style="margin-top:8px;">
                    Configure batch options and click <b>Run Batch</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #1e3a5f;margin:30px 0 10px 0;">
<div style="text-align:center;color:#3a5a7a;font-size:0.75em;letter-spacing:1px;">
    RADAR THREAT CLASSIFICATION SYSTEM &nbsp;|&nbsp;
    HB100 24 GHz &amp; SAAB SIRS 77 GHz &nbsp;|&nbsp;
    IDP 2026
</div>
""", unsafe_allow_html=True)
