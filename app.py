import streamlit as st
import joblib
from collections import Counter

# Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# Loading Models and Vectorizer
# Use st.cache_resource to load models only once
@st.cache_resource
def load_models_and_vectorizer():
    try:
        models = {
            "Logistic Regression": joblib.load('logistic_regression.joblib'),
            "Decision Tree": joblib.load('decision_tree.joblib'),
            "Gradient Boosting": joblib.load('gradient_boosting.joblib'),
            "Random Forest": joblib.load('random_forest.joblib')
        }
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return models, vectorizer
    except FileNotFoundError:
        return None, None


models, vectorizer = load_models_and_vectorizer()

# Custom CSS for Styling
st.markdown("""
<style>
    .main {
        background-color: #F5F5F5;
    }
    .stTextArea textarea {
        height: 300px;
        font-size: 1.1rem;
        border: 2px solid #1E90FF;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #1E90FF;
        color: white;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    h1, .stMarkdown {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main Application
st.title("Advanced Fake News Detector")
st.markdown("Enter the text of a news article below to check its authenticity.")

if not models or not vectorizer:
    st.error("Models not found. Please run the training script first to generate the model files.")
else:
    # Centered text area
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        text_input = st.text_area("News Article Text", "")
        predict_button = st.button("Analyze News")

    if predict_button and text_input:
        st.write("---")
        st.header("Analysis Results")

        # Preprocessing the input text
        text_tfidf = vectorizer.transform([text_input])

        # Getting predictions from all models
        predictions = []

        # Displaying individual model predictions
        col1, col2 = st.columns(2)
        model_names = list(models.keys())

        for i, name in enumerate(model_names):
            model = models[name]
            prediction_val = model.predict(text_tfidf)[0]
            prediction_text = "Fake News" if prediction_val == 1 else "True News"
            predictions.append(prediction_text)

            # Distributing results into two columns
            if i % 2 == 0:
                with col1:
                    st.subheader(name)
                    st.info(f"Prediction: **{prediction_text}**")
            else:
                with col2:
                    st.subheader(name)
                    st.info(f"Prediction: **{prediction_text}**")

        st.write("---")

        # Final Prediction Based on Majority Vote
        st.header("Final Verdict")

        # Calculate the most common prediction (majority vote)
        final_prediction = Counter(predictions).most_common(1)[0][0]

        if final_prediction == "Fake News":
            st.error(f"**Final Verdict:** The models suggest this is likely **{final_prediction}**.")
        else:
            st.success(f"**Final Verdict:** The models suggest this is likely **{final_prediction}**.")