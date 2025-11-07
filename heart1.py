"""
Heart1 - Advanced Cardiac Health Studio
Focused, streamlined Streamlit app for cardiovascular monitoring, CVD risk assessment,
and electromechanical heart simulation. This app intentionally does not include
Blood Pressure prediction to keep the scope tight and responsive.

Unique features vs dual_model_health_app:
- Patient profile selector (per-session patient context)
- One-click demo wearable stream (auto-refresh option)
- Notes & goals journal for patient follow-up
- CSV export for CVD assessments and wearable data
"""

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Heart1 - Cardiac Health Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💗"
)

import numpy as np
import pandas as pd
import sqlite3
import os
import json
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv('med.env')

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CVD_FEATURE_COLUMNS = ['age', 'cp', 'thalach', 'oldpeak', 'thal']
CVD_PIPELINE_FILE = 'cardio.joblib'

# ----------------------------------------------------------------------------
# DB
# ----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect('heart1.db', check_same_thread=False)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS wearable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            patient_id TEXT,
            heart_rate INTEGER,
            hrv REAL,
            steps INTEGER,
            spo2 REAL,
            activity TEXT,
            raw_json TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS cvd (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            patient_id TEXT,
            age INTEGER,
            cp INTEGER,
            thalach REAL,
            oldpeak REAL,
            thal INTEGER,
            risk_score REAL,
            category TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            patient_id TEXT,
            text TEXT
        )
    ''')
    conn.commit()
    return conn

DB = init_db()

def save_wearable(patient_id: str, data: dict):
    cur = DB.cursor()
    cur.execute(
        'INSERT INTO wearable (ts, patient_id, heart_rate, hrv, steps, spo2, activity, raw_json) VALUES (?,?,?,?,?,?,?,?)',
        (
            data['ts'], patient_id, data['hr'], data['hrv'], data['steps'], data['spo2'], data['activity'],
            json.dumps(data.get('raw', {}))
        )
    )
    DB.commit()

def save_cvd(patient_id: str, payload: dict):
    cur = DB.cursor()
    cur.execute(
        'INSERT INTO cvd (ts, patient_id, age, cp, thalach, oldpeak, thal, risk_score, category) VALUES (?,?,?,?,?,?,?,?,?)',
        (
            payload['ts'], patient_id, payload['age'], payload['cp'], payload['thalach'],
            payload['oldpeak'], payload['thal'], payload['risk_score'], payload['category']
        )
    )
    DB.commit()

def save_note(patient_id: str, text: str):
    cur = DB.cursor()
    cur.execute('INSERT INTO notes (ts, patient_id, text) VALUES (?,?,?)', (datetime.now().isoformat(), patient_id, text))
    DB.commit()

def get_wearable(patient_id: str, limit: int = 100):
    cur = DB.cursor()
    cur.execute('SELECT ts, heart_rate, hrv, steps, spo2, activity FROM wearable WHERE patient_id=? ORDER BY ts DESC LIMIT ?', (patient_id, limit))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=['Timestamp', 'Heart Rate', 'HRV', 'Steps', 'SpO2', 'Activity']) if rows else pd.DataFrame()

def get_cvd(patient_id: str, limit: int = 100):
    cur = DB.cursor()
    cur.execute('SELECT ts, age, cp, thalach, oldpeak, thal, risk_score, category FROM cvd WHERE patient_id=? ORDER BY ts DESC LIMIT ?', (patient_id, limit))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=['Timestamp', 'Age', 'CP', 'Thalach', 'Oldpeak', 'Thal', 'Risk Score', 'Category']) if rows else pd.DataFrame()

# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
@st.cache_resource
def load_cvd_pipeline(path: str):
    try:
        pipe = joblib.load(path)
        return pipe, True
    except Exception:
        # light placeholder
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(200, 5)
        y = np.random.randint(0, 2, 200)
        clf.fit(X, y)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        return pipe, False

CVD_PIPELINE, CVD_READY = load_cvd_pipeline(CVD_PIPELINE_FILE)

def predict_cvd(age, cp, thalach, oldpeak, thal):
    X = pd.DataFrame([[age, cp, thalach, oldpeak, thal]], columns=CVD_FEATURE_COLUMNS)
    try:
        proba = CVD_PIPELINE.predict_proba(X)[0]
        p = float(proba[1]) if len(proba) == 2 else float(proba[0])
    except Exception:
        p = float(np.random.uniform(0.2, 0.8))
    score = round(p * 100)
    if p >= 0.75:
        cat, color, icon = "CRITICAL RISK", "#dc2626", "🔴"
    elif p >= 0.65:
        cat, color, icon = "HIGH RISK", "#ef4444", "🟠"
    elif p >= 0.45:
        cat, color, icon = "MODERATE RISK", "#f59e0b", "🟡"
    elif p >= 0.25:
        cat, color, icon = "LOW-MODERATE RISK", "#84cc16", "🟢"
    else:
        cat, color, icon = "LOW RISK", "#10b981", "✅"
    return score, cat, color, icon

# ----------------------------------------------------------------------------
# SIDEBAR - Patient context
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("💗 Heart1 Studio")
    st.caption("Focused toolkit for cardiac health workups")
    st.markdown("---")
    patient_id = st.text_input("Patient ID", value=st.session_state.get('patient_id', 'patient_A'))
    st.session_state['patient_id'] = patient_id
    demo_stream = st.checkbox("Auto demo wearable stream", value=True)
    st.markdown("---")
    st.write("Model status:")
    st.success("CVD model loaded" if CVD_READY else "CVD placeholder active")

st.markdown("## Heart1 - Cardiac Health Studio")
st.markdown("Robust monitoring, risk assessment and simulation — BP prediction intentionally omitted.")
st.markdown("---")

# Tabs: Wearables, CVD Risk, Heart Simulation, Monitoring, Analytics, Journal
tab_wear, tab_cvd, tab_sim, tab_rt, tab_ana, tab_journal = st.tabs([
    "⌚ Wearables", "❤️ CVD Risk", "🫀 Simulation", "📡 Monitoring", "📊 Analytics", "📝 Journal"
])

# ----------------------------------------------------------------------------
# Wearables
# ----------------------------------------------------------------------------
with tab_wear:
    # Monitoring controls integrated here
    st.subheader("Heart Rate Monitor")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Live watch readings are fetched automatically on each load.")
    with c2:
        st.empty()

    # Always create a fresh reading on reload
    data = {
        'ts': datetime.now().isoformat(),
        'hr': int(np.random.randint(60, 100)),
        'hrv': float(np.random.uniform(20, 60)),
        'steps': int(np.random.randint(0, 12000)),
        'spo2': float(np.random.uniform(95, 100)),
        'activity': np.random.choice(['resting', 'light', 'moderate', 'vigorous']),
        'raw': {'ecg': np.random.randn(100).tolist()}
    }
    st.session_state['wearable_latest'] = data
    save_wearable(patient_id, data)

    sub_overview, sub_trend = st.tabs(["Overview", "Trend & Details"])

    with sub_overview:
        st.subheader("Live Watch Feed")
        if st.button("Generate Watch Reading"):
            data = {
                'ts': datetime.now().isoformat(),
                'hr': int(np.random.randint(60, 100)),
                'hrv': float(np.random.uniform(20, 60)),
                'steps': int(np.random.randint(0, 12000)),
                'spo2': float(np.random.uniform(95, 100)),
                'activity': np.random.choice(['resting', 'light', 'moderate', 'vigorous']),
                'raw': {'ecg': np.random.randn(100).tolist()}
            }
            st.session_state['wearable_latest'] = data
            save_wearable(patient_id, data)
        latest = st.session_state.get('wearable_latest')
        if latest:
            colm1, colm2, colm3, colm4 = st.columns(4)
            colm1.metric("Heart Rate", f"{latest['hr']} BPM")
            colm2.metric("HRV", f"{latest['hrv']:.1f} ms")
            colm3.metric("SpO2", f"{latest['spo2']:.1f}%")
            colm4.metric("Steps", f"{latest['steps']:,}")
            st.caption(f"Activity: {latest['activity']} • {latest['ts']}")
        else:
            st.info("Waiting for first reading…")

    with sub_trend:
        st.subheader("Recent Trend")
        df_w = get_wearable(patient_id, 100)
        if not df_w.empty:
            df_w['Timestamp'] = pd.to_datetime(df_w['Timestamp'])
            df_w = df_w.sort_values('Timestamp')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_w['Timestamp'], y=df_w['Heart Rate'], mode='lines+markers', name='HR', line=dict(color='#ef4444')))
            fig.add_trace(go.Scatter(x=df_w['Timestamp'], y=df_w['SpO2'], mode='lines+markers', name='SpO2', yaxis='y2', line=dict(color='#3b82f6')))
            fig.update_layout(
                height=300,
                yaxis=dict(title='HR (BPM)'),
                yaxis2=dict(title='SpO2 (%)', overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Export Wearable CSV", data=df_w.to_csv(index=False), file_name=f"{patient_id}_wearable.csv")
        else:
            st.info("No wearable history yet.")

# ----------------------------------------------------------------------------
# CVD Risk
# ----------------------------------------------------------------------------
with tab_cvd:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Risk Inputs")
        # Use latest HR from wearable if present
        w = st.session_state.get('wearable_latest')
        thalach_default = int(w['hr']) if w else int(np.random.randint(70, 190))
        age = st.slider("Age", 29, 80, 48)
        cp = st.selectbox("Chest Pain Type", options=[0,1,2,3], format_func=lambda x: ['Typical','Atypical','Non-Anginal','Asymptomatic'][x])
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=210, value=thalach_default)
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        thal = st.selectbox("Thallium Test (thal)", options=[1,2,3], format_func=lambda x: {1:'Normal',2:'Fixed Defect',3:'Reversible Defect'}[x])
        if st.button("Assess CVD Risk", use_container_width=True):
            score, cat, color, icon = predict_cvd(age, cp, thalach, oldpeak, thal)
            payload = {
                'ts': datetime.now().isoformat(), 'age': age, 'cp': cp, 'thalach': thalach,
                'oldpeak': oldpeak, 'thal': thal, 'risk_score': score, 'category': cat
            }
            st.session_state['cvd_result'] = payload
            save_cvd(patient_id, payload)
    with col2:
        st.subheader("Result")
        res = st.session_state.get('cvd_result')
        if res:
            st.metric("Risk Score", f"{res['risk_score']}%")
            st.markdown(f"**Category:** {res['category']}")
        else:
            st.info("Fill inputs and click Assess CVD Risk")
        df_c = get_cvd(patient_id, 20)
        if not df_c.empty:
            st.markdown("---")
            st.dataframe(df_c[['Timestamp','Risk Score','Category']], use_container_width=True, hide_index=True)
            st.download_button("⬇️ Export CVD CSV", data=df_c.to_csv(index=False), file_name=f"{patient_id}_cvd.csv")

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
with tab_sim:
    st.subheader("Electromechanical Heart Model (lightweight demo)")
    hr = st.slider("Resting HR (BPM)", 50, 120, 72)
    sv = st.slider("Stroke Volume (mL)", 40, 120, 70)
    co = (hr * sv) / 1000.0
    map_est = round(co * 18.0, 1)  # simple illustrative factor
    st.metric("Cardiac Output", f"{co:.2f} L/min")
    st.metric("Mean Arterial Pressure (est)", f"{map_est} mmHg")
    x = np.arange(0, 6.0, 0.01)
    p_wave = 0.1*np.exp(-((x%1.0-0.1)**2)/(2*0.05**2))
    qrs = 0.8*np.exp(-((x%1.0-0.2)**2)/(2*0.02**2))
    t_wave = 0.3*np.exp(-((x%1.0-0.4)**2)/(2*0.1**2))
    ecg = p_wave + qrs - t_wave
    fig = go.Figure(go.Scatter(x=x[:600], y=ecg[:600], line=dict(color="#ef4444", width=1)))
    fig.update_layout(height=250, title="ECG-like Signal (demo)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Monitoring
# ----------------------------------------------------------------------------
with tab_rt:
    st.subheader("Recent Alerts (demo)")
    df_w = get_wearable(patient_id, 30)
    if not df_w.empty:
        alerts = []
        for _, r in df_w.iterrows():
            if r['Heart Rate'] > 100:
                alerts.append((r['Timestamp'], 'Heart Rate', r['Heart Rate'], 'warning', 'Tachycardia trend'))
            if r['SpO2'] < 95:
                alerts.append((r['Timestamp'], 'SpO2', r['SpO2'], 'warning', 'Low oxygen saturation'))
        if alerts:
            for a in alerts[-10:]:
                st.markdown(f"- [{a[0]}] {a[1]}: {a[2]} ({a[3]}) — {a[4]}")
        else:
            st.success("No alert-worthy events detected")
    else:
        st.info("No wearable data yet")

# ----------------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------------
with tab_ana:
    st.subheader("Overview")
    df_w = get_wearable(patient_id, 200)
    df_c = get_cvd(patient_id, 200)
    if not df_w.empty:
        df_w['Timestamp'] = pd.to_datetime(df_w['Timestamp'])
        fig = px.line(df_w.sort_values('Timestamp'), x='Timestamp', y='Heart Rate', title='Heart Rate Trend', color_discrete_sequence=['#ef4444'])
        st.plotly_chart(fig, use_container_width=True)
    if not df_c.empty:
        df_c['Timestamp'] = pd.to_datetime(df_c['Timestamp'])
        fig2 = px.line(df_c.sort_values('Timestamp'), x='Timestamp', y='Risk Score', title='CVD Risk Score Trend', color_discrete_sequence=['#7c3aed'])
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# Journal
# ----------------------------------------------------------------------------
with tab_journal:
    st.subheader("Notes & Goals")
    note = st.text_area("Add a note for this patient", placeholder="Diet, exercise, medication adherence, follow-up…")
    colj1, colj2 = st.columns([1,1])
    with colj1:
        if st.button("Save Note") and note.strip():
            save_note(patient_id, note.strip())
            st.success("Saved")
    with colj2:
        st.caption("Export available from data tables above")

# ----------------------------------------------------------------------------
# Style (distinct from dual app)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main { background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }
      .stApp { color: #111827; }
      h1, h2, h3, h4 { color: #111827 !important; }
      .stMetric { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; }
      .stMetric [data-testid="stMetricLabel"],
      .stMetric [data-testid="stMetricDelta"],
      .stMetric [data-testid="stMetricValue"] { color: #111827 !important; }
      .stMetric [data-testid="stMetricValue"] { font-weight: 800 !important; }
      .stTabs [data-baseweb="tab"] { background: #ffffff; border: 1px solid #e5e7eb; color:#111827 !important; }
      .stTabs [data-baseweb="tab"]:hover { background:#f3f4f6; }
    </style>
    """,
    unsafe_allow_html=True,
)

"""
DTwin: AI-Powered Digital Health Twin
A comprehensive cardiovascular health monitoring and risk assessment platform.
"""

# ============================================================================
# IMPORTS & DEPENDENCIES
# ============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import json
import requests
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple

# Load environment variables
load_dotenv('med.env')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
FEATURE_COLUMNS = ['age', 'cp', 'thalach', 'oldpeak', 'thal']
PIPELINE_FILE = 'cardio.joblib'
DB_FILE = 'heart_health.db'

# ============================================================================
# DATABASE MANAGER
# ============================================================================
class DatabaseManager:
    """Manages all database operations for heart health data."""
    
    def __init__(self, db_file: str):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                age INTEGER,
                cp INTEGER,
                thalach REAL,
                oldpeak REAL,
                thal INTEGER,
                risk_score REAL,
                risk_category TEXT,
                session_id TEXT,
                health_score REAL
            )
        ''')
        
        # Check if health_score column exists, if not add it
        try:
            cursor.execute("PRAGMA table_info(assessments)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'health_score' not in columns:
                cursor.execute('ALTER TABLE assessments ADD COLUMN health_score REAL DEFAULT 0')
                self.conn.commit()
        except Exception as e:
            pass
        
        # Heart rate sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS heart_rate_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration REAL,
                avg_heart_rate REAL,
                max_heart_rate REAL,
                min_heart_rate REAL,
                hrv REAL,
                data_points TEXT
            )
        ''')
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                created_at TEXT
            )
        ''')
        
        self.conn.commit()
    
    def save_assessment(self, assessment_data: Dict) -> int:
        """Save assessment to database and return assessment ID."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO assessments 
            (timestamp, age, cp, thalach, oldpeak, thal, risk_score, risk_category, session_id, health_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            assessment_data.get('timestamp', datetime.now().isoformat()),
            assessment_data.get('age'),
            assessment_data.get('cp'),
            assessment_data.get('thalach'),
            assessment_data.get('oldpeak'),
            assessment_data.get('thal'),
            assessment_data.get('risk_score'),
            assessment_data.get('risk_category'),
            assessment_data.get('session_id', ''),
            assessment_data.get('health_score', 0)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_assessment_history(self, limit: int = 100) -> List[Tuple]:
        """Retrieve assessment history from database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM assessments 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()
    
    def get_statistics(self) -> Dict:
        """Get aggregated statistics from assessments."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*), AVG(risk_score), AVG(health_score) FROM assessments')
        result = cursor.fetchone()
        return {
            'total_assessments': result[0] or 0,
            'avg_risk_score': result[1] or 0,
            'avg_health_score': result[2] or 0
        }
    
    def clear_history(self):
        """Clear all assessment history."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM assessments')
        self.conn.commit()

# Initialize database
db_manager = DatabaseManager(DB_FILE)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize all session state variables."""
    if 'heartrate_data' not in st.session_state:
        st.session_state.heartrate_data = {
            'time': deque(maxlen=200),
            'rate': deque(maxlen=200),
            'is_monitoring': False,
            'start_time': None,
            'average_rate': None,
            'session_id': str(time.time())
        }
    
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Guest User',
            'age': 45,
            'gender': 'Not specified'
        }

initialize_session_state()

# ============================================================================
# ML PIPELINE FUNCTIONS
# ============================================================================
@st.cache_resource
def load_pipeline(filename: str) -> Pipeline:
    """Loads the saved ML pipeline."""
    try:
        pipeline = joblib.load(filename)
        return pipeline
    except FileNotFoundError:
        st.error(f"⚠️ Error: Pipeline file '{filename}' not found.")
        st.stop()
    except Exception as e:
        # Compatibility fix for pipelines pickled with different scikit-learn versions
        try:
            import sklearn.compose._column_transformer as _ct
            if not hasattr(_ct, '_RemainderColsList'):
                class _RemainderColsList(list):
                    pass
                _ct._RemainderColsList = _RemainderColsList
            pipeline = joblib.load(filename)
            return pipeline
        except Exception:
            # Fallback: create a lightweight placeholder pipeline so the app remains usable
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            X = np.random.rand(200, 5)
            y = np.random.randint(0, 2, 200)
            clf.fit(X, y)
            placeholder = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            st.warning("Loaded placeholder model due to incompatible pipeline. Re-train or export with current scikit-learn to avoid this.")
            return placeholder

def get_dtwin_risk_output(raw_input_data: List, pipeline: Pipeline) -> Dict:
    """Enhanced risk assessment with detailed breakdown."""
    raw_array = np.array(raw_input_data).reshape(1, -1) 
    live_df = pd.DataFrame(raw_array, columns=FEATURE_COLUMNS)
    proba = pipeline.predict_proba(live_df)[0, 0] 
    
    risk_probability = float(proba)
    risk_percentage = round(risk_probability * 100)
    
    # Enhanced risk categorization
    if risk_probability >= 0.75:
        category = "CRITICAL RISK"
        color = "#dc2626"
        icon = "🔴"
    elif risk_probability >= 0.65:
        category = "HIGH RISK"
        color = "#ef4444"
        icon = "🟠"
    elif risk_probability >= 0.45:
        category = "MODERATE RISK"
        color = "#f59e0b"
        icon = "🟡"
    elif risk_probability >= 0.25:
        category = "LOW-MODERATE RISK"
        color = "#84cc16"
        icon = "🟢"
    else:
        category = "LOW RISK"
        color = "#10b981"
        icon = "✅"
    
    # Individual factor analysis
    age, cp, thalach, oldpeak, thal = raw_input_data
    factors = {
        'Age': {
            'value': age, 
            'risk': 'High' if age > 60 else 'Moderate' if age > 50 else 'Low',
            'normal_range': '30-70',
            'description': 'Age factor in cardiovascular risk'
        },
        'Chest Pain': {
            'value': cp, 
            'risk': 'High' if cp == 3 else 'Moderate' if cp >= 1 else 'Low',
            'normal_range': '0-3',
            'description': 'Chest pain type indicator'
        },
        'Max Heart Rate': {
            'value': thalach, 
            'risk': 'High' if thalach < 120 else 'Moderate' if thalach < 150 else 'Low',
            'normal_range': '120-180',
            'description': 'Maximum heart rate achieved'
        },
        'ST Depression': {
            'value': oldpeak, 
            'risk': 'High' if oldpeak > 2.0 else 'Moderate' if oldpeak > 1.0 else 'Low',
            'normal_range': '0-1.0',
            'description': 'ST depression measurement'
        },
        'Thallium Test': {
            'value': thal, 
            'risk': 'High' if thal == 3 else 'Moderate' if thal == 2 else 'Low',
            'normal_range': '1-3',
            'description': 'Thallium stress test result'
        }
    }
    
    return {
        "risk_category": category,
        "risk_score_percent": risk_percentage,
        "raw_probability": risk_probability,
        "color": color,
        "icon": icon,
        "factors": factors,
        "timestamp": datetime.now().isoformat()
    }

def calculate_health_score(risk_data: Dict, heart_data: Dict) -> Dict:
    """Calculate comprehensive health score."""
    base_score = 100 - risk_data['risk_score_percent']
    
    # Heart rate bonus/penalty
    hr = heart_data.get('average_rate')
    if hr:
        if 60 <= hr <= 100:
            hr_score = 10
        elif 50 <= hr < 60 or 100 < hr <= 110:
            hr_score = 5
        else:
            hr_score = -10
    else:
        hr_score = 0
    
    # HRV bonus
    hrv = np.std(list(heart_data['rate'])) if heart_data.get('rate', []) else None
    if hrv and 20 <= hrv <= 50:
        hrv_score = 5
    else:
        hrv_score = 0
    
    total_score = max(0, min(100, base_score + hr_score + hrv_score))
    
    if total_score >= 80:
        grade = "Excellent"
    elif total_score >= 65:
        grade = "Good"
    elif total_score >= 50:
        grade = "Fair"
    else:
        grade = "Needs Attention"
    
    return {
        'score': total_score,
        'grade': grade,
        'breakdown': {
            'base': base_score,
            'heart_rate': hr_score,
            'hrv': hrv_score
        }
    }

# ============================================================================
# AI RECOMMENDATIONS FUNCTION
# ============================================================================
def get_health_recommendations(risk_data: Dict, heart_data: Dict) -> Dict:
    """Get AI-powered health recommendations based on real-time health metrics."""
    OPENROUTER_KEY = os.getenv('OPENROUTER_KEY')
    if not OPENROUTER_KEY:
        return {"success": False, "error": "OpenRouter API key not found in environment variables."}

    heart_rate = heart_data.get('average_rate', None)
    heart_variability = np.std(list(heart_data['rate'])) if heart_data.get('rate', []) else None
    risk_level = risk_data['risk_category']
    risk_score = risk_data['risk_score_percent']
    
    # Factor analysis summary
    factors_summary = "\n".join([
        f"- {name}: {info['value']} (Risk: {info['risk']})"
        for name, info in risk_data.get('factors', {}).items()
    ])

    system_prompt = """You are an advanced AI cardiology assistant providing comprehensive health guidance.
    Structure your response in these sections:

    📊 VITAL SIGNS INTERPRETATION:
    - Analyze current heart rate and variability
    - Compare to healthy baselines
    - Identify concerning patterns
    
    🎯 PERSONALIZED RECOMMENDATIONS:
    - Immediate actions (next 24 hours)
    - Short-term goals (next week)
    - Long-term lifestyle changes
    
    📈 RISK FACTOR BREAKDOWN:
    - Address each specific risk factor
    - Provide targeted interventions
    
    ⚠️ WARNING SIGNS TO MONITOR:
    - Specific symptoms to watch for
    - When to seek immediate medical attention
    
    Use clear, actionable language with specific examples."""

    heart_rate_str = f"{heart_rate:.1f} BPM" if heart_rate is not None else "Not monitored"
    heart_var_str = f"{heart_variability:.2f} ms" if heart_variability is not None else "Not available"
    
    user_context = f"""PATIENT ASSESSMENT:
    Overall Risk: {risk_level} ({risk_score}%)
    
    Individual Risk Factors:
    {factors_summary}
    
    Real-time Vitals:
    - Heart Rate: {heart_rate_str}
    - Heart Rate Variability: {heart_var_str}
    
    Provide comprehensive health guidance based on this assessment."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer": "https://github.com/",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3-haiku",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            recommendations = response.json()["choices"][0]["message"]["content"]
            return {"success": True, "recommendations": recommendations}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_heart_rate_chart(heart_data: Dict) -> go.Figure:
    """Create real-time heart rate visualization."""
    fig = go.Figure()
    
    if len(heart_data['rate']) > 0:
        fig.add_trace(go.Scatter(
            x=list(heart_data['time']),
            y=list(heart_data['rate']),
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#764ba2'),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.15)'
        ))
        
        # Add reference zones
        fig.add_hrect(y0=60, y1=100, fillcolor="rgba(16, 185, 129, 0.15)", 
                     layer="below", line_width=0, annotation_text="Normal Zone",
                     annotation_position="top left")
    
    fig.update_layout(
        title={'text': '<b>Real-Time Heart Rate Monitoring</b>', 'x': 0.5, 'font': {'size': 18, 'color': '#1a1a1a'}},
        xaxis_title='<b>Time (seconds)</b>',
        yaxis_title='<b>Heart Rate (BPM)</b>',
        height=350,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        font=dict(color='#1a1a1a', size=12),
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    return fig

def create_risk_factor_chart(assessment: Dict) -> go.Figure:
    """Create risk factor visualization."""
    factors_df = pd.DataFrame({
        'Factor': list(assessment['factors'].keys()),
        'Risk Level': [info['risk'] for info in assessment['factors'].values()],
        'Value': [info['value'] for info in assessment['factors'].values()]
    })
    
    risk_map = {'High': 3, 'Moderate': 2, 'Low': 1}
    factors_df['Risk Score'] = factors_df['Risk Level'].map(risk_map)
    
    fig = px.bar(
        factors_df,
        x='Factor',
        y='Risk Score',
        color='Risk Level',
        color_discrete_map={'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#10b981'},
        title="<b>Individual Risk Factor Analysis</b>",
        labels={'Risk Score': 'Risk Level', 'Factor': 'Health Factor'}
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        font=dict(color='#1a1a1a', size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_risk_distribution_chart(assessment: Dict) -> go.Figure:
    """Create risk distribution pie chart."""
    factors_df = pd.DataFrame({
        'Factor': list(assessment['factors'].keys()),
        'Risk Level': [info['risk'] for info in assessment['factors'].values()]
    })
    
    risk_counts = factors_df['Risk Level'].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color_discrete_map={'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#10b981'},
        hole=0.4,
        title="<b>Risk Factor Distribution</b>"
    )
    fig.update_layout(
        height=400,
        font=dict(color='#1a1a1a', size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
    return fig

def create_risk_trend_chart(history_df: pd.DataFrame) -> go.Figure:
    """Create risk trend over time chart."""
    if len(history_df) == 0:
        return None
    
    history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
    history_df = history_df.sort_values('Timestamp')
    
    fig = px.line(
        history_df,
        x='Timestamp',
        y='Risk Score',
        markers=True,
        title="<b>Risk Score Trend Over Time</b>",
        color_discrete_sequence=['#667eea']
    )
    fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig.add_hline(y=35, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
    fig.update_layout(
        height=400,
        font=dict(color='#1a1a1a', size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_health_score_gauge(health_score: Dict) -> go.Figure:
    """Create circular gauge for health score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score['score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Health Score</b>", 'font': {'size': 20, 'color': '#1a1a1a'}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': '#667eea'},
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 65], 'color': '#fef3c7'},
                {'range': [65, 80], 'color': '#d1fae5'},
                {'range': [80, 100], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, font=dict(color='#1a1a1a', size=14))
    return fig

# ============================================================================

# ============================================================================
# ENHANCED CUSTOM CSS STYLING - MAXIMUM VISIBILITY & ATTRACTIVENESS
# ============================================================================
st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* Global styles - ULTRA DARK TEXT for maximum visibility */
    * {
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Force all text to be very dark */
    h1, h2, h3, h4, h5, h6, p, div, span, label, li, td, th {
        color: #000000 !important;
    }
    
    /* Main container with BRIGHT LIGHT background */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%) !important;
        background-attachment: fixed;
    }
    
    /* Force white/light background everywhere */
    body {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%) !important;
    }
    
    /* Force all containers to have light backgrounds */
    div[data-testid="stAppViewContainer"] {
        background: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
    }
    
    /* Headers - Very dark with strong contrast */
    .main-header {
        color: #000000 !important;
        font-weight: 900;
        font-size: 3.5em;
        text-align: center;
        padding: 25px 0;
        letter-spacing: -1px;
        text-shadow: 2px 2px 8px rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, #1e293b 0%, #000000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: #000000;
        background-clip: text;
    }
    
    .sub-header {
        color: #000000 !important;
        font-weight: 700;
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 30px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Card containers with BRIGHT WHITE background */
    .card-container {
        background: #ffffff !important;
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 24px;
        border: 2px solid #e2e8f0 !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.2);
    }
    
    .card-title {
        color: #000000 !important;
        font-size: 1.8em;
        font-weight: 800;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .card-subtitle {
        color: #1a1a1a !important;
        font-size: 1.1em;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    /* Info box with light background */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
        border: 2px solid #3b82f6;
    }
    
    .info-box h4 {
        color: #000000 !important;
        font-weight: 800;
        margin-bottom: 12px;
        font-size: 1.5em;
    }
    
    .info-box p {
        color: #000000 !important;
        line-height: 1.9;
        margin: 0;
        font-size: 1.1em;
        font-weight: 600;
    }
    
    /* Buttons - Enhanced styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 56px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-size: 1.1em;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Start/Stop Monitoring Button - Special styling */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4) !important;
        font-weight: 800 !important;
        font-size: 1.2em !important;
        color: #ffffff !important;
    }
    
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.5) !important;
    }
    
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4) !important;
        font-weight: 800 !important;
        font-size: 1.2em !important;
        color: #ffffff !important;
    }
    
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.5) !important;
    }
    
    /* Selectbox styling - Comprehensive override */
    .stSelectbox {
        background-color: transparent !important;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 10px !important;
        min-height: 40px !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.05em !important;
    }
    
    /* Selectbox focus state */
    .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Selectbox text color - all text elements */
    .stSelectbox span,
    .stSelectbox p,
    .stSelectbox div {
        color: #000000 !important;
    }
    
    /* Baseweb select component override - comprehensive */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 10px !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="select"] input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="select"]:hover {
        border-color: #667eea !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    
    /* Select value text */
    [data-baseweb="select"] span {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Selectbox popover/dropdown */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important;
    }
    
    /* Selectbox list items */
    [data-baseweb="popover"] [role="listbox"] li {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
    }
    
    [data-baseweb="popover"] [role="listbox"] li:hover {
        background-color: #eff6ff !important;
        color: #1e40af !important;
    }
    
    [data-baseweb="popover"] [role="listbox"] li[aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    
    /* Additional selectbox overrides */
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Status indicators */
    .status-recording {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    
    .status-standby {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(107, 114, 128, 0.3);
    }
    
    .status-text {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 1.1em;
        letter-spacing: 0.5px;
    }
    
    /* Metrics cards */
    .metric-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-label {
        font-size: 0.95em;
        margin: 0;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .metric-value {
        margin: 12px 0 0 0;
        font-size: 3em;
        font-weight: 800;
        line-height: 1;
    }
    
    /* Risk assessment cards */
    .risk-card-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 45px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(239, 68, 68, 0.4);
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    .risk-card-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 45px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(245, 158, 11, 0.4);
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    .risk-card-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 45px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.4);
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    .risk-title {
        color: white;
        font-size: 1.4em;
        font-weight: 700;
        margin-bottom: 25px;
        letter-spacing: 0.8px;
    }
    
    .risk-percentage {
        color: white;
        font-size: 5.5em;
        font-weight: 800;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.3);
        line-height: 1;
        display: block;
        margin: 25px 0;
    }
    
    .risk-category {
        background: rgba(255,255,255,0.25);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.4);
    }
    
    .risk-category-text {
        color: white;
        font-size: 2.2em;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 0;
    }
    
    /* Input labels - ULTRA DARK for maximum visibility */
    .stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.1em !important;
        text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2);
    }
    
    /* Section headers - Very dark with visible colored border */
    .section-header {
        color: #000000 !important;
        font-size: 1.3em;
        font-weight: 800;
        margin: 28px 0 18px 0;
        padding-bottom: 12px;
        border-bottom: 4px solid #3b82f6 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
    }
    
    /* Custom horizontal rule styling - Make separators visible */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%);
        margin: 30px 0;
        border-radius: 2px;
    }
    
    /* Streamlit horizontal rule override */
    .stMarkdown hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%) !important;
        margin: 30px 0 !important;
        border-radius: 2px;
    }
    
    /* Recommendations container with bright background */
    .recommendations-container {
        background: #ffffff !important;
        padding: 32px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 24px;
        border: 3px solid #3b82f6 !important;
    }
    
    .recommendations-title {
        color: #000000 !important;
        font-size: 2em;
        font-weight: 900;
        text-align: center;
        margin-bottom: 28px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Recommendation sections with light background */
    .rec-section {
        background: #ffffff !important;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        border-left: 5px solid;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .rec-section h3 {
        color: #000000 !important;
        font-size: 1.5em;
        font-weight: 800;
        margin-bottom: 18px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
    }
    
    .rec-section p, .rec-section div {
        color: #000000 !important;
        line-height: 2.0;
        font-weight: 600;
        font-size: 1.1em;
    }
    
    .rec-section-immediate {
        border-left-color: #667eea;
    }
    
    .rec-section-risk {
        border-left-color: #ef4444;
    }
    
    .rec-section-vitals {
        border-left-color: #10b981;
    }
    
    /* Factor pills */
    .factor-pill {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        margin: 6px;
        font-size: 0.95em;
        font-weight: 700;
    }
    
    .factor-high {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .factor-moderate {
        background: #fef3c7;
        color: #92400e;
    }
    
    .factor-low {
        background: #d1fae5;
        color: #065f46;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Text content - ULTRA DARK for maximum visibility */
    p, div, span, li, td, th, strong, b, em {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Streamlit text elements */
    .stMarkdown, .stText, .stMetric, .stDataFrame {
        color: #000000 !important;
    }
    
    /* Table text */
    table {
        color: #000000 !important;
    }
    
    table td, table th {
        color: #000000 !important;
        font-weight: 700;
    }
    
    /* Info boxes text */
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #000000 !important;
    }
    
    /* Make all Streamlit text dark */
    .element-container {
        color: #000000 !important;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Additional visibility enhancements */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.15em !important;
    }
    
    /* Chart titles and labels */
    .js-plotly-plot .gtitle {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load pipeline
DTWIN_PIPELINE = load_pipeline(PIPELINE_FILE)

# ============================================================================
# MAIN APPLICATION LAYOUT
# ============================================================================

# Header Section
st.markdown('<h1 class="main-header">🫀 DTwin: AI-Powered Digital Health Twin</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time Cardiovascular Risk Assessment & Personalized Health Monitoring</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="info-box">
        <h4>👋 Welcome to Your Digital Health Twin</h4>
        <p>Monitor your heart health in real-time and receive personalized AI-powered recommendations. 
        Start by monitoring your heart rate or entering your health metrics manually for instant risk assessment.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard", "📈 Historical Trends", "📊 Analytics & Visualizations", "⚙️ Settings"
])

# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================
with tab1:
    # Reorganized layout - Better structure
    # Top row: Heart Rate Monitor and Patient Info side by side
    col1, col2 = st.columns([0.0001, 1])
    
    # COLUMN 1: HEART RATE MONITORING
    with col1:
        st.empty()
    
    # COLUMN 2: PATIENT DATA INPUT
    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📋 Patient Information</h3>', unsafe_allow_html=True)
        st.markdown('<p class="card-subtitle">Enter your health metrics</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">👤 Demographics</p>', unsafe_allow_html=True)
        age = st.slider("Age", 29, 77, st.session_state.user_profile.get('age', 45), key='age', 
                       help="Your current age in years")
        
        st.markdown('<p class="section-header">💔 Chest Pain Assessment</p>', unsafe_allow_html=True)
        cp_map = {
            0: "😣 Typical Angina (High Risk)",
            1: "😕 Atypical Angina (Moderate)",
            2: "😌 Non-Anginal Pain (Low)",
            3: "😐 Asymptomatic (No Pain)"
        }
        cp_value = st.selectbox("Chest Pain Type", options=list(cp_map.keys()), 
                              format_func=lambda x: cp_map[x], key='cp',
                              help="Select the type of chest pain you experience")
        
        st.markdown('<p class="section-header">❤️ Cardiac Metrics</p>', unsafe_allow_html=True)
        
        # Use monitored heart rate if available
        if st.session_state.heartrate_data['average_rate'] is not None:
            thalach = st.session_state.heartrate_data['average_rate']
            st.success(f"✅ Using monitored heart rate: **{thalach:.1f} BPM**")
        else:
            thalach = st.slider("Maximum Heart Rate", 70, 202, 150, key='thalach',
                              help="Highest heart rate achieved during exercise")
        
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 0.0, step=0.1, key='oldpeak',
                           help="ST depression induced by exercise relative to rest")
        
        st.markdown('<p class="section-header">🔬 Thallium Stress Test</p>', unsafe_allow_html=True)
        thal_map = {
            1: "✓ Normal Blood Flow",
            2: "⚠ Fixed Defect (Scar)",
            3: "⚠ Reversible Defect"
        }
        thal_value = st.selectbox("Test Result", options=list(thal_map.keys()), 
                                 format_func=lambda x: thal_map[x], key='thal',
                                 help="Result from thallium stress test")
        
        st.markdown("<br>", unsafe_allow_html=True)
        raw_input = [age, cp_value, thalach, oldpeak, thal_value]
        predict_button = st.button("🔍 Calculate Risk Score", type="primary", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk Assessment Section - Below the two columns
    st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
    if predict_button or st.session_state.current_assessment:
        if predict_button:
            # Get risk assessment
            assessment = get_dtwin_risk_output(raw_input, DTWIN_PIPELINE)
            assessment['age'] = age
            assessment['cp'] = cp_value
            assessment['thalach'] = thalach
            assessment['oldpeak'] = oldpeak
            assessment['thal'] = thal_value
            
            # Calculate health score
            health_score = calculate_health_score(assessment, st.session_state.heartrate_data)
            assessment['health_score'] = health_score['score']
            
            # Save to database
            assessment['risk_score'] = assessment['risk_score_percent']
            db_manager.save_assessment(assessment)
            
            st.session_state.current_assessment = assessment
            st.session_state.current_health_score = health_score
            st.rerun()
        else:
            assessment = st.session_state.current_assessment
            health_score = st.session_state.current_health_score
        
        # Display Risk Assessment and Health Score side by side
        col_assess1, col_assess2 = st.columns([1, 1])
        
        with col_assess1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown(f'<h3 class="card-title">{assessment["icon"]} Risk Assessment</h3>', unsafe_allow_html=True)
            
            # Health Score Gauge
            health_gauge = create_health_score_gauge(health_score)
            st.plotly_chart(health_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_assess2:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            # Risk display
            risk_class = "risk-card-high" if "HIGH" in assessment['risk_category'] or "CRITICAL" in assessment['risk_category'] else \
                        "risk-card-medium" if "MODERATE" in assessment['risk_category'] else "risk-card-low"
            
            st.markdown(f"""
                <div class="{risk_class}">
                    <h3 class="risk-title">CARDIOVASCULAR RISK ASSESSMENT</h3>
                    <span class="risk-percentage">{assessment['risk_score_percent']}%</span>
                    <div class="risk-category">
                        <p class="risk-category-text">{assessment['risk_category']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">🎯 Risk Assessment</h3>', unsafe_allow_html=True)
        st.markdown('<p class="card-subtitle">AI-powered cardiovascular analysis</p>', unsafe_allow_html=True)
        st.info("👆 Enter your health metrics and click 'Calculate Risk Score' to get started")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # AI Recommendations Section
    if st.session_state.current_assessment:
        st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-container recommendations-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="recommendations-title">🤖 AI Health Insights & Recommendations</h2>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Recommendations", use_container_width=True):
            st.session_state.loading_recommendations = True
        
        if 'loading_recommendations' in st.session_state or 'recommendations' not in st.session_state:
            with st.spinner("🤖 AI is analyzing your health data..."):
                recommendation_result = get_health_recommendations(
                    st.session_state.current_assessment,
                    st.session_state.heartrate_data
                )
                
                if recommendation_result.get("success", False):
                    st.session_state.recommendations = recommendation_result["recommendations"]
                    st.session_state.loading_recommendations = False
        
        if 'recommendations' in st.session_state:
            # Parse recommendations
            recommendations = st.session_state.recommendations
            try:
                sections = recommendations.split("🎯 PERSONALIZED RECOMMENDATIONS:")
                vitals_section = sections[0].replace("📊 VITAL SIGNS INTERPRETATION:", "").strip()
                
                if len(sections) > 1:
                    remaining = sections[1].split("📈 RISK FACTOR BREAKDOWN:")
                    immediate_section = remaining[0].strip()
                    
                    if len(remaining) > 1:
                        risk_remaining = remaining[1].split("⚠️ WARNING SIGNS TO MONITOR:")
                        risk_section = risk_remaining[0].strip()
                        warning_section = risk_remaining[1].strip() if len(risk_remaining) > 1 else ""
                    else:
                        risk_section = ""
                        warning_section = ""
                else:
                    immediate_section = ""
                    risk_section = ""
                    warning_section = ""
            except:
                vitals_section = recommendations
                immediate_section = ""
                risk_section = ""
                warning_section = ""
            
            # Vital Signs Analysis
            if vitals_section:
                st.markdown(f"""
                    <div class="rec-section rec-section-vitals">
                        <h3>📊 Vital Signs Interpretation</h3>
                        <div style="color: #000000 !important; line-height: 2.0; font-weight: 700; font-size: 1.1em;">
                            {vitals_section.replace(chr(10), '<br>')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Personalized Recommendations
            if immediate_section:
                st.markdown(f"""
                    <div class="rec-section rec-section-immediate">
                        <h3>🎯 Personalized Recommendations</h3>
                        <div style="color: #000000 !important; line-height: 2.0; font-weight: 700; font-size: 1.1em;">
                            {immediate_section.replace(chr(10), '<br>')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Risk Factor Breakdown
            if risk_section:
                st.markdown(f"""
                    <div class="rec-section rec-section-risk">
                        <h3>📈 Risk Factor Breakdown</h3>
                        <div style="color: #000000 !important; line-height: 2.0; font-weight: 700; font-size: 1.1em;">
                            {risk_section.replace(chr(10), '<br>')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Warning Signs
            if warning_section:
                st.markdown(f"""
                    <div class="rec-section rec-section-risk">
                        <h3>⚠️ Warning Signs to Monitor</h3>
                        <div style="color: #000000 !important; line-height: 2.0; font-weight: 700; font-size: 1.1em;">
                            {warning_section.replace(chr(10), '<br>')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Diet Plans & Suggestions Section
    st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">🥗 Personalized Diet Plans & Health Suggestions</h3>', unsafe_allow_html=True)
    st.markdown('<p class="card-subtitle">Customized nutrition and lifestyle recommendations based on your health profile</p>', unsafe_allow_html=True)
    
    # Diet plans based on risk level
    if st.session_state.current_assessment:
        assessment = st.session_state.current_assessment
        risk_level = assessment['risk_category']
        risk_score = assessment['risk_score_percent']
        
        # Determine diet plan category
        if "HIGH" in risk_level or "CRITICAL" in risk_level:
            diet_category = "Cardiac-Protective Diet"
            diet_color = "#ef4444"
            diet_icon = "🔴"
        elif "MODERATE" in risk_level:
            diet_category = "Heart-Healthy Balanced Diet"
            diet_color = "#f59e0b"
            diet_icon = "🟡"
        else:
            diet_category = "Maintenance Wellness Diet"
            diet_color = "#10b981"
            diet_icon = "🟢"
        
        col_diet1, col_diet2 = st.columns(2)
        
        with col_diet1:
            st.markdown(f"""
                <div style="padding: 24px; margin: 12px 0; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 16px; border-left: 5px solid {diet_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h3 style="color: #000000 !important; font-weight: 800; font-size: 1.4em; margin-bottom: 15px;">
                        {diet_icon} Recommended Diet Plan: {diet_category}
                    </h3>
                    <div style="color: #000000 !important; line-height: 1.9; font-weight: 600;">
                        <p><strong>✅ Foods to Include:</strong></p>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Omega-3 rich fish (salmon, mackerel, sardines)</li>
                            <li>Fresh fruits and vegetables (5-7 servings daily)</li>
                            <li>Whole grains (oats, brown rice, quinoa)</li>
                            <li>Nuts and seeds (walnuts, almonds, flaxseeds)</li>
                            <li>Lean proteins (chicken, turkey, legumes)</li>
                            <li>Low-fat dairy products</li>
                        </ul>
                        <p style="margin-top: 15px;"><strong>❌ Foods to Limit:</strong></p>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Processed foods high in sodium</li>
                            <li>Saturated and trans fats</li>
                            <li>Added sugars and sweetened beverages</li>
                            <li>Excessive alcohol consumption</li>
                            <li>Red meat (limit to 2-3 times per week)</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_diet2:
            st.markdown(f"""
                <div style="padding: 24px; margin: 12px 0; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 16px; border-left: 5px solid #10b981; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h3 style="color: #000000 !important; font-weight: 800; font-size: 1.4em; margin-bottom: 15px;">
                        💪 Lifestyle Recommendations
                    </h3>
                    <div style="color: #000000 !important; line-height: 1.9; font-weight: 600;">
                        <p><strong>📅 Daily Routine:</strong></p>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>30-45 minutes of moderate exercise</li>
                            <li>7-8 hours of quality sleep</li>
                            <li>Morning meditation or deep breathing (10-15 min)</li>
                            <li>Stay hydrated (8-10 glasses of water)</li>
                        </ul>
                        <p style="margin-top: 15px;"><strong>🎯 Weekly Goals:</strong></p>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>150 minutes of aerobic activity</li>
                            <li>2-3 strength training sessions</li>
                            <li>Stress management activities</li>
                            <li>Regular health check-ups</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Meal Planning Suggestions
        st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #000000 !important; font-weight: 800; font-size: 1.3em; margin: 20px 0 15px 0;">🍽️ Sample Meal Plan (Daily)</h4>', unsafe_allow_html=True)
        
        meal_col1, meal_col2, meal_col3 = st.columns(3)
        
        with meal_col1:
            st.markdown("""
                <div style="padding: 20px; background: #ffffff; border-radius: 12px; border: 2px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <h5 style="color: #000000 !important; font-weight: 800; font-size: 1.1em; margin-bottom: 12px;">🌅 Breakfast</h5>
                    <p style="color: #000000 !important; font-weight: 600; line-height: 1.8;">
                        • Oatmeal with berries & nuts<br>
                        • Greek yogurt with honey<br>
                        • Green tea or herbal tea
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with meal_col2:
            st.markdown("""
                <div style="padding: 20px; background: #ffffff; border-radius: 12px; border: 2px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <h5 style="color: #000000 !important; font-weight: 800; font-size: 1.1em; margin-bottom: 12px;">🌞 Lunch</h5>
                    <p style="color: #000000 !important; font-weight: 600; line-height: 1.8;">
                        • Grilled salmon or chicken<br>
                        • Mixed green salad<br>
                        • Quinoa or brown rice<br>
                        • Fresh fruit
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with meal_col3:
            st.markdown("""
                <div style="padding: 20px; background: #ffffff; border-radius: 12px; border: 2px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <h5 style="color: #000000 !important; font-weight: 800; font-size: 1.1em; margin-bottom: 12px;">🌙 Dinner</h5>
                    <p style="color: #000000 !important; font-weight: 600; line-height: 1.8;">
                        • Lean protein (fish/poultry)<br>
                        • Steamed vegetables<br>
                        • Whole grain or sweet potato<br>
                        • Light soup or broth
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Additional Tips
        st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #000000 !important; font-weight: 800; font-size: 1.3em; margin: 20px 0 15px 0;">💡 Additional Health Tips</h4>', unsafe_allow_html=True)
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
                <div style="padding: 20px; background: #fff7ed; border-radius: 12px; border-left: 5px solid #f59e0b;">
                    <h5 style="color: #000000 !important; font-weight: 800; font-size: 1.1em; margin-bottom: 12px;">⚖️ Portion Control</h5>
                    <p style="color: #000000 !important; font-weight: 600; line-height: 1.8;">
                        • Use smaller plates to control portions<br>
                        • Fill half your plate with vegetables<br>
                        • Eat slowly and mindfully<br>
                        • Stop eating when 80% full
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with tips_col2:
            st.markdown("""
                <div style="padding: 20px; background: #f0fdf4; border-radius: 12px; border-left: 5px solid #10b981;">
                    <h5 style="color: #000000 !important; font-weight: 800; font-size: 1.1em; margin-bottom: 12px;">🧘 Stress Management</h5>
                    <p style="color: #000000 !important; font-weight: 600; line-height: 1.8;">
                        • Practice deep breathing exercises<br>
                        • Regular meditation or yoga<br>
                        • Maintain social connections<br>
                        • Get adequate sleep (7-8 hours)
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("💡 Complete a risk assessment first to receive personalized diet plans and suggestions based on your health profile.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 2: HISTORICAL TRENDS (between Dashboard and Analytics)
# ============================================================================
with tab2:
    st.markdown("<h2 style='color: #000000 !important; font-weight: 800; font-size: 2em; margin-bottom: 20px;'>📈 Historical Trends & Analysis</h2>", unsafe_allow_html=True)
    history = db_manager.get_assessment_history(100)
    if len(history) == 0:
        st.info("📊 No assessment history yet. Complete assessments to see trends here.")
    else:
        history_df = pd.DataFrame(history, columns=['ID','Timestamp','Age','CP','Thalach','Oldpeak','Thal','Risk Score','Risk Category','Session ID','Health Score'])
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        history_df = history_df.sort_values('Timestamp')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assessments", len(history_df))
        with col2:
            st.metric("Average Risk Score", f"{history_df['Risk Score'].mean():.1f}%")
        with col3:
            st.metric("Average Health Score", f"{history_df['Health Score'].mean():.1f}")
        with col4:
            latest_risk = history_df['Risk Score'].iloc[-1]
            prev_risk = history_df['Risk Score'].iloc[-2] if len(history_df) > 1 else latest_risk
            delta = latest_risk - prev_risk
            st.metric("Latest Risk Score", f"{latest_risk:.1f}%", f"{delta:+.1f}%")
        st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title">📈 Risk Score Trend</h3>', unsafe_allow_html=True)
            risk_trend_chart = create_risk_trend_chart(history_df)
            if risk_trend_chart:
                st.plotly_chart(risk_trend_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title">📊 Recent Assessments</h3>', unsafe_allow_html=True)
            recent_df = history_df[['Timestamp','Risk Score','Risk Category','Health Score']].tail(10)
            recent_df['Timestamp'] = recent_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            recent_df.columns = ['Date & Time','Risk Score (%)','Risk Category','Health Score']
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        csv = history_df.to_csv(index=False)
        st.download_button("⬇️ Export Full History (CSV)", data=csv, file_name="cvd_history.csv")

# ============================================================================
# TAB 3: ANALYTICS & VISUALIZATIONS
# ============================================================================
with tab3:
    st.markdown("<h2 style='color: #000000 !important; font-weight: 800; font-size: 2em; margin-bottom: 20px;'>📊 Advanced Health Analytics & Visualizations</h2>", unsafe_allow_html=True)
    if st.session_state.current_assessment:
        assessment = st.session_state.current_assessment
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title">📈 Risk Factor Impact Analysis</h3>', unsafe_allow_html=True)
            risk_factor_chart = create_risk_factor_chart(assessment)
            st.plotly_chart(risk_factor_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title">🎯 Risk Distribution</h3>', unsafe_allow_html=True)
            risk_dist_chart = create_risk_distribution_chart(assessment)
            st.plotly_chart(risk_dist_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">🔍 Detailed Risk Factor Breakdown</h3>', unsafe_allow_html=True)
        for factor_name, factor_info in assessment['factors'].items():
            risk_class = 'factor-high' if factor_info['risk'] == 'High' else 'factor-moderate' if factor_info['risk'] == 'Moderate' else 'factor-low'
            st.markdown(f"""
                <div style="padding: 22px; margin: 14px 0; background: #ffffff; border-radius: 16px; border-left: 5px solid #667eea; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-weight: 800; color: #000000 !important; font-size: 1.2em; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">{factor_name}</span>
                        <span class="factor-pill {risk_class}">{factor_info['risk']} Risk</span>
                    </div>
                    <p style="margin: 10px 0; color: #000000 !important; font-weight: 700; font-size: 1.05em;">Value: <strong style="color: #000000 !important; font-weight: 900;">{factor_info['value']}</strong></p>
                    <p style="margin: 6px 0; color: #000000 !important; font-weight: 600; font-size: 1em;">Normal Range: <strong>{factor_info.get('normal_range', 'N/A')}</strong></p>
                    <p style="margin: 6px 0; color: #000000 !important; font-weight: 600; font-size: 0.95em;">{factor_info.get('description', '')}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please complete a risk assessment first to view analytics")

# ============================================================================
# TAB 4: SETTINGS
# ============================================================================
with tab4:
    st.markdown("<h2 style='color: #000000 !important; font-weight: 800; font-size: 2em; margin-bottom: 20px;'>⚙️ Application Settings & User Profile</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">👤 User Profile</h3>', unsafe_allow_html=True)
        
        name = st.text_input("Name", value=st.session_state.user_profile['name'])
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
        age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.user_profile['age'])
        
        if st.button("💾 Save Profile", use_container_width=True):
            st.session_state.user_profile.update({
                'name': name,
                'gender': gender,
                'age': age
            })
            st.success("✅ Profile saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📊 Statistics</h3>', unsafe_allow_html=True)
        
        stats = db_manager.get_statistics()
        st.metric("Total Assessments", stats['total_assessments'])
        st.metric("Average Risk Score", f"{stats['avg_risk_score']:.1f}%")
        st.metric("Average Health Score", f"{stats['avg_health_score']:.1f}")
        
        st.markdown('<hr style="border: none; height: 3px; background: linear-gradient(90deg, #3b82f6 0%, #667eea 50%, #764ba2 100%); margin: 30px 0; border-radius: 2px;">', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Assessment History", use_container_width=True):
            db_manager.clear_history()
            st.success("✅ History cleared successfully!")
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
