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

    if not query or query.isspace():
        return jsonify({"error": "No search query provided"}), 400

    query = re.sub(r"\W+", " ", query).strip().lower()

    results = df[
        df["title"].str.lower().str.contains(query, case=False, na=False, regex=False)
    ]
    if results.empty:
        return jsonify({"message": "No books found"}), 404
    search_results = results.to_dict(orient="records")
    return jsonify(search_results)

