# 📧 Spam Detector AI

A modern, premium spam detection application built with **Streamlit** and **Naive Bayes**.

## 🚀 Live Demo
The app is deployed on Streamlit Cloud.

## 🛠️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prithivirj4706/naive_bayes.git
   cd naive_bayes
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python model.py
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Streamlit frontend.
- `model.py`: Training script to generate `model.pkl` and `vectorizer.pkl`.
- `spam.csv`: The dataset used for training.
- `requirements.txt`: List of Python dependencies.

## 🧠 Model
The app uses a **Multinomial Naive Bayes** classifier with a `CountVectorizer` to process text data.
