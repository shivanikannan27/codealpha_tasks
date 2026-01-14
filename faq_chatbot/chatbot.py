import string
import nltk
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

print("🤖 FAQ Chatbot is ready! (type 'exit' to quit)\n")

# Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    # Preprocess user question
    user_processed = preprocess(user_input)

    # Vectorize user question
    user_vector = vectorizer.transform([user_processed])

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, faq_vectors)

    # Get best matching FAQ
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    # Confidence threshold
    if best_score < 0.2:
        print("Bot: Sorry, I couldn't understand your question.")
    else:
        print("Bot:", faqs[best_match_index]["answer"])

