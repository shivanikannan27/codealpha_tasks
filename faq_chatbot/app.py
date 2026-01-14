import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import faqs

# Load stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Prepare FAQ questions
questions = [preprocess(faq["question"]) for faq in faqs]

# Vectorize questions
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

# Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")
st.write("Ask a question related to the product or service.")

user_input = st.text_input("You:")

if user_input:
    user_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_processed])
    similarities = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.2:
        st.error("Sorry, I couldn't understand your question.")
    else:
        st.success(f"**Bot:** {faqs[best_match_index]['answer']}")
