import streamlit as st
import requests

API_URL = "https://sexism-detector-1000036994845.europe-west1.run.app/predict"

st.set_page_config(page_title="Sexism Detector", page_icon="⚡")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .rainbow {
        background: linear-gradient(90deg, #ff0000, #ff8800, #ffff00, #00cc00, #0088ff, #8800ff, #ff0088);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .big { font-size: 3.5rem; text-align: center; margin-bottom: 0.5rem; }
    .prob-box {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-top: 1.5rem;
    }
    .prob-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff0088, #ff8800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .label-sexist { color: #ff4444; font-size: 2rem; font-weight: 700; }
    .label-safe   { color: #44ff88; font-size: 2rem; font-weight: 700; }
    .stTextArea textarea { background: #1e1e2e; color: #eee; border: 1px solid #444; border-radius: 12px; }
    .stTextArea textarea:focus { border-color: #ff0088; box-shadow: 0 0 12px rgba(255,0,136,0.3); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="rainbow big">⚡ Sexism Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#aaa;margin-bottom:2rem;">Ensemble: DistilBERT + Logistic Regression</p>', unsafe_allow_html=True)

text = st.text_area(
    "Enter text to analyze",
    height=140,
    placeholder="Paste or type a sentence here...",
)

if text:
    with st.spinner("Analyzing..."):
        response = requests.post(API_URL, json={"text": text})
        result = response.json()

    prob_pct = result["confidence"] * 100
    label = result["label"]
    is_sexist = label == "sexist"

    st.markdown('<div class="prob-box">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#aaa;margin:0;">Probability of sexism</p>'
        f'<p class="prob-value">{prob_pct:.1f}%</p>'
        f'<p class="label-{"sexist" if is_sexist else "safe"}">'
        f'{"⚠️ SEXIST" if is_sexist else "✅ NOT SEXIST"}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="prob-box" style="opacity:0.4;">'
        '<p style="color:#666;font-size:1.2rem;margin:0;">Awaiting input...</p>'
        "</div>",
        unsafe_allow_html=True,
    )
