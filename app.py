import streamlit as st
import pickle

# Set page configuration
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="📧",
    layout="centered"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #1a1c24;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #3d4150;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    .result-container {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .spam {
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
    }
    .ham {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid #28a745;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("📧 Spam Detector AI")
st.markdown("Enter an email or message below to check if it's **Spam** or **Ham** using Naive Bayes.")

# Load Model and Vectorizer
@st.cache_resource
def load_assets():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.info("Try running 'model.py' locally to generate correct pickle files for your sklearn version.")
    st.stop()

# User Input
message = st.text_area("Message Content", placeholder="Type your message here...", height=150)

if st.button("Analyze Message"):
    if message.strip():
        # Vectorize input
        data = [message]
        vect = vectorizer.transform(data)
        
        # Prediction
        prediction = model.predict(vect)[0]
        
        # Display Result
        if prediction == 'spam':
            st.markdown(f'<div class="result-container spam">🚨 This is a SPAM message!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-container ham">✅ This is a HAM (Genuine) message.</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")


# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit & Naive Bayes</div>", unsafe_allow_html=True)
