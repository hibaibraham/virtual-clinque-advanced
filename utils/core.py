"""
Utilitaires partagés — chargement modèle, CSS, helpers.
"""
import os, json, re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR     = os.path.join(BASE_DIR, 'saved_models')
DATA_PATH    = os.path.join(BASE_DIR, 'thyroid.csv')
HISTORY_PATH = os.path.join(BASE_DIR, 'prediction_history.csv')
CSS_PATH     = os.path.join(BASE_DIR, 'style.css')

NORMAL_RANGES = {
    'TSH': (0.4, 4.0),
    'T3':  (1.2, 3.1),
    'TT4': (70,  180),
    'FTI': (70,  180),
    'T4U': (0.7, 1.3),
}

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': '#94a3b8'},
    title_font_color='#f1f5f9',
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    legend=dict(font=dict(color='#94a3b8')),
)


@st.cache_resource
def load_css() -> str:
    with open(CSS_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def inject_css():
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


def inject_bg():
    st.markdown("""
<div id="bg-particles">
  <div class="particle" style="width:90px;height:90px;background:rgba(0,212,255,0.12);left:4%;animation-duration:20s;animation-delay:0s;"></div>
  <div class="particle" style="width:55px;height:55px;background:rgba(124,58,237,0.15);left:14%;animation-duration:15s;animation-delay:2s;"></div>
  <div class="particle" style="width:130px;height:130px;background:rgba(0,212,255,0.07);left:24%;animation-duration:25s;animation-delay:4s;"></div>
  <div class="particle" style="width:40px;height:40px;background:rgba(16,185,129,0.15);left:36%;animation-duration:17s;animation-delay:1s;"></div>
  <div class="particle" style="width:100px;height:100px;background:rgba(124,58,237,0.1);left:50%;animation-duration:22s;animation-delay:3s;"></div>
  <div class="particle" style="width:65px;height:65px;background:rgba(0,212,255,0.1);left:63%;animation-duration:16s;animation-delay:5s;"></div>
  <div class="particle" style="width:110px;height:110px;background:rgba(16,185,129,0.08);left:74%;animation-duration:26s;animation-delay:0.5s;"></div>
  <div class="particle" style="width:48px;height:48px;background:rgba(124,58,237,0.12);left:84%;animation-duration:18s;animation-delay:2.5s;"></div>
  <div class="dna-dot" style="left:8%;top:18%;animation-delay:0s;"></div>
  <div class="dna-dot" style="left:22%;top:62%;animation-delay:0.7s;"></div>
  <div class="dna-dot" style="left:42%;top:33%;animation-delay:1.4s;"></div>
  <div class="dna-dot" style="left:58%;top:72%;animation-delay:2.1s;"></div>
  <div class="dna-dot" style="left:76%;top:22%;animation-delay:0.3s;"></div>
  <div class="dna-dot" style="left:91%;top:58%;animation-delay:1.8s;"></div>
</div>
<svg id="heartbeat-line" height="60" viewBox="0 0 1440 60" preserveAspectRatio="none">
  <polyline class="ecg-path" fill="none" stroke="#00d4ff" stroke-width="2"
    points="0,30 80,30 100,30 110,5 120,55 130,10 145,50 160,30 240,30
            320,30 340,30 350,5 360,55 370,10 385,50 400,30 480,30
            560,30 580,30 590,5 600,55 610,10 625,50 640,30 720,30
            800,30 820,30 830,5 840,55 850,10 865,50 880,30 960,30
            1040,30 1060,30 1070,5 1080,55 1090,10 1105,50 1120,30 1200,30
            1280,30 1300,30 1310,5 1320,55 1330,10 1345,50 1360,30 1440,30"/>
</svg>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model        = joblib.load(os.path.join(SAVE_DIR, 'model.joblib'))
    preprocessor = joblib.load(os.path.join(SAVE_DIR, 'preprocessor.joblib'))
    with open(os.path.join(SAVE_DIR, 'feature_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    return model, preprocessor, config


@st.cache_data(show_spinner=False, ttl=300)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def compute_engineered_features(row: dict) -> dict:
    row['TSH_abnormal'] = int((row.get('TSH', 0) < 0.4)  or (row.get('TSH', 0) > 4.0))
    row['TT4_abnormal'] = int((row.get('TT4', 0) < 70)   or (row.get('TT4', 0) > 180))
    row['T3_abnormal']  = int((row.get('T3',  0) < 1.2)  or (row.get('T3',  0) > 3.1))
    row['FTI_abnormal'] = int((row.get('FTI', 0) < 70)   or (row.get('FTI', 0) > 180))
    row['hormone_score']  = row['TSH_abnormal'] + row['TT4_abnormal'] + row['T3_abnormal'] + row['FTI_abnormal']
    row['T4U_TT4_ratio']  = row.get('T4U', 0) / (row.get('TT4', 0) + 1e-6)
    return row


def save_prediction(patient_data: dict, prediction: int, probability: float):
    from utils.firebase import save_prediction_firestore
    # Firebase
    fb_record = save_prediction_firestore(patient_data, prediction, probability)
    # Local CSV fallback
    record = {
        'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': 'Pathologique' if prediction == 1 else 'Normal',
        'probability': f"{probability:.1%}",
        **{k: v for k, v in patient_data.items()}
    }
    df_new = pd.DataFrame([record])
    if os.path.exists(HISTORY_PATH):
        df_hist = pd.read_csv(HISTORY_PATH)
        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df_hist = df_new
    df_hist.to_csv(HISTORY_PATH, index=False)
    return fb_record if fb_record else record


def confidence_badge(prob: float) -> str:
    if prob >= 0.80 or prob <= 0.20:
        level, label = 'high',   'Confiance Élevée'
    elif prob >= 0.65 or prob <= 0.35:
        level, label = 'medium', 'Confiance Moyenne'
    else:
        level, label = 'low',    'Confiance Faible'
    return f'<span class="confidence-badge confidence-{level}">● {label}</span>'


def input_indicator(value: float, marker: str) -> str:
    lo, hi = NORMAL_RANGES.get(marker, (None, None))
    if lo is None:
        return ''
    ok = lo <= value <= hi
    cls = 'input-ok' if ok else 'input-bad'
    tip = 'Normal' if ok else f'Hors norme ({lo}–{hi})'
    return f'<span class="input-indicator {cls}" title="{tip}"></span>'


def extract_label(val) -> str:
    if pd.isna(val):
        return 'Inconnu'
    match = re.match(r'^([^\[]+)', str(val))
    return match.group(1).strip() if match else str(val)


def page_header(tag: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="main-header">
        <div class="header-tag">{tag}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="pro-footer">
        <span class="status-dot"></span>
        MedAI Thyroid v3.0 &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; Streamlit &amp; scikit-learn
        <br><span style='font-size:0.72rem;opacity:0.6;'>Outil d'aide à la décision — ne remplace pas un avis médical professionnel</span>
    </div>""", unsafe_allow_html=True)
