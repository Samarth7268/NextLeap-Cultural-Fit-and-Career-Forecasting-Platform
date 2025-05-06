from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

new = pd.read_csv("C:/Users/91636/OneDrive/Documents/Culture Match Using NLP/merged_data.csv")

custom_stop_words = [word for word in TfidfVectorizer(stop_words='english').get_stop_words() if word not in ['no', 'not']]
vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
tfidf_matrix = vectorizer.fit_transform(new["Text"])

def recommend_companies(user_input, top_n=5):
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return new.iloc[top_indices][["Company Name"]]

user_input = "I am a Data Analyst , looking for a Business Consulting firm with high package and good work-life balance "
recommendations = recommend_companies(user_input)
print(recommendations)
