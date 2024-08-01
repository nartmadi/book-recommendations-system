from flask import request, jsonify, current_app as app, abort
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity

tfidf = joblib.load("models/tfidf_vectorizer_eng.pkl")
scaler = joblib.load("models/scaler.pkl")
combined_features = np.load("models/combined_features.npy")
df = pd.read_csv("models/books_data.csv")


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")

    if not query:
        return jsonify({"error": "No search query provided"}), 400

    query = re.sub(r"\W+", " ", query).strip().lower()

    results = df[
        df["title"].str.lower().str.contains(query, case=False, na=False, regex=False)
    ]
    if results.empty:
        return jsonify({"message": "No books found"}), 404
    search_results = results.to_dict(orient="records")
    return jsonify(search_results)


@app.route("/recommend", methods=["POST"])
def recommend():
    liked_books = request.json.get("liked_books", [])
    if not liked_books:
        abort(400, description="No liked books provided")

    liked_indices = df[df["title"].isin(liked_books)].index.tolist()
    if not liked_indices:
        abort(404, description="Liked books not found in database")

    user_profile = np.mean(combined_features[liked_indices], axis=0)
    cosine_sim = cosine_similarity([user_profile], combined_features)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_books = [
        df["title"].iloc[i[0]]
        for i in sim_scores
        if df["title"].iloc[i[0]] not in liked_books
    ]

    return jsonify(recommended_books[:10])

liked_books = []

@app.route('/likes', methods=['POST'])
def add_like():
    book_title = request.json.get('title')
    if not book_title:
        return jsonify({"error": "No book title provided"}), 400

    if book_title not in liked_books:
        liked_books.append(book_title)
        return jsonify({"message": "Book added to likes"}), 200
    else:
        return jsonify({"message": "Book already in likes"}), 200
