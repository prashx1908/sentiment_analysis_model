import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import shap
import joblib

# Function to clean text data
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove hyperlinks
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(subset=['LABEL'], inplace=True)  # Remove rows with NaN in LABEL
    data = data[data['LABEL'].isin(['positive', 'negative'])]  # Ensure LABEL only contains 'positive' and 'negative'
    data['cleaned_caption'] = data['Caption'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_caption']).toarray()
    y = data['LABEL'].map({'positive': 1, 'negative': 0})
    return X, y, vectorizer, data

# Train and evaluate the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report, X_test, y_test, y_pred

# Generate word clouds
def generate_wordcloud(data, label):
    text = ' '.join(data[data['LABEL'] == label]['cleaned_caption'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# App layout
def main():
    st.title("SentiX: Sentiment Analysis Model")

    # Load and preprocess data
    file_path = 'senti.csv'
    X, y, vectorizer, data = load_and_preprocess_data(file_path)

    # Train the model
    model, report, X_test, y_test, y_pred = train_model(X, y)
    
    st.subheader("Model Performance")
    st.text("Precision, Recall, F1-score, and Support for each class")
    st.write(pd.DataFrame(report).transpose())
    
    st.subheader("Test the Emotions")
    user_input = st.text_area("Enter a statement to classify its sentiment")
    if st.button("Predict Sentiment"):
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input]).toarray()
        prediction = model.predict(input_vector)
        sentiment = "Positive Statement, relates to good events or happeneing" if prediction[0] == 1 else "Negative Statement, relates to having bad events or happening"
        st.write(f"Predicted Sentiment: {sentiment}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    st.pyplot(fig)

    # Word clouds
    st.subheader("Word Clouds")
    st.text("Word Cloud for Positive Sentiments")
    wordcloud_pos = generate_wordcloud(data, 'positive')
    fig, ax = plt.subplots()
    ax.imshow(wordcloud_pos, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.text("Word Cloud for Negative Sentiments")
    wordcloud_neg = generate_wordcloud(data, 'negative')
    fig, ax = plt.subplots()
    ax.imshow(wordcloud_neg, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input_vector)
    st.subheader("Feature Contribution to Prediction")
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value, shap_values, feature_names=vectorizer.get_feature_names_out()))


    

if __name__ == "__main__":
    main()
