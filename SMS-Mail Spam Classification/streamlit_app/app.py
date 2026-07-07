import streamlit as st
import pickle
import string
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------------
# NLTK SETUP
# On Hugging Face Spaces the container is fresh each build, so we download the
# required NLTK corpora at runtime instead of relying on them being pre-installed.
# ----------------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

setup_nltk()

ps = PorterStemmer()

# ----------------------------------------------------------------------------------
# TEXT PREPROCESSING
# NOTE: This must be IDENTICAL to the preprocessing you used before fitting
# the TF-IDF vectorizer during training, or predictions will be unreliable.
# ----------------------------------------------------------------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ----------------------------------------------------------------------------------
# LOAD MODEL & VECTORIZER
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    tfidf, model = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ----------------------------------------------------------------------------------
# CUSTOM CSS — makes the default Streamlit look a lot less "default"
# ----------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    }
    .title-text {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #b0b3c1;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.8rem;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #3d3d5c;
        font-size: 1rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white;
        border: none;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 18px rgba(0, 114, 255, 0.45);
    }
    .result-card {
        padding: 1.4rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 1.2rem;
        animation: fadeIn 0.4s ease-in;
    }
    .spam-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
    }
    .ham-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: #0b2b1f;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .footer-note {
        text-align: center;
        color: #7a7d91;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ About")
    st.write(
        "This app uses a **TF-IDF vectorizer** and a trained **machine learning model** "
        "to classify SMS/text messages as **Spam** or **Not Spam (Ham)**."
    )
    st.markdown("---")
    st.markdown("### 🔍 How it works")
    st.write(
        "1. Enter or pick a message\n"
        "2. Text is cleaned, tokenized & stemmed\n"
        "3. TF-IDF converts it to numeric features\n"
        "4. The model predicts the class"
    )
    st.markdown("---")
    st.markdown("### 📊 Session stats")
    if "total_checked" not in st.session_state:
        st.session_state.total_checked = 0
        st.session_state.total_spam = 0
    st.metric("Messages checked", st.session_state.total_checked)
    st.metric("Flagged as spam", st.session_state.total_spam)
    st.markdown("---")
    st.caption("Built with Streamlit • Deployed on Hugging Face Spaces")

# ----------------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------------
st.markdown('<p class="title-text">📩 SMS Spam Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle-text">Paste any text message below and let the model decide — Spam or Not Spam.</p>',
    unsafe_allow_html=True,
)

if not artifacts_loaded:
    st.error(
        "⚠️ Could not find **model.pkl** and/or **vectorizer.pkl** in the app directory. "
        "Make sure both files are uploaded to the root of this Space."
    )
    st.stop()

# ----------------------------------------------------------------------------------
# EXAMPLE MESSAGES
# ----------------------------------------------------------------------------------
st.markdown("**Try an example:**")
example_col1, example_col2, example_col3 = st.columns(3)

examples = {
    "🎉 Prize scam": "Congratulations! You have WON a $1000 Walmart gift card. Click here to claim now!!!",
    "👋 Casual text": "Hey, are we still on for lunch tomorrow at 1pm?",
    "📞 Loan spam": "URGENT! You have been selected for a pre-approved loan. Call now to claim your cash prize.",
}

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

for col, (label, text) in zip([example_col1, example_col2, example_col3], examples.items()):
    if col.button(label):
        st.session_state.message_input = text

# ----------------------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------------------
message = st.text_area(
    "Enter the message here 👇",
    value=st.session_state.message_input,
    height=140,
    placeholder="e.g. Congratulations! You've been selected to win a free prize...",
    key="message_input",
)

predict_clicked = st.button("🔍 Check Message")

# ----------------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------------
if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(0.4)  # small delay purely for UX polish
            transformed = transform_text(message)
            vector_input = tfidf.transform([transformed])
            prediction = model.predict(vector_input)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector_input)[0]
                confidence = round(max(proba) * 100, 2)

        st.session_state.total_checked += 1

        is_spam = int(prediction) == 1
        if is_spam:
            st.session_state.total_spam += 1
            st.markdown(
                f'<div class="result-card spam-card">🚨 SPAM DETECTED'
                + (f'<br><span style="font-size:1rem;font-weight:500;">Confidence: {confidence}%</span>' if confidence else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.balloons()
        else:
            st.markdown(
                f'<div class="result-card ham-card">✅ NOT SPAM (Safe Message)'
                + (f'<br><span style="font-size:1rem;font-weight:500;">Confidence: {confidence}%</span>' if confidence else "")
                + "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("🔧 See the preprocessed text fed into the model"):
            st.code(transformed if transformed else "(empty after preprocessing)")

st.markdown(
    '<p class="footer-note">Model & vectorizer loaded locally from this Space • '
    "Predictions are for demonstration purposes only.</p>",
    unsafe_allow_html=True,
)
