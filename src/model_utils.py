# src/model_utils.py

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GerVADER.vaderSentimentGER import SentimentIntensityAnalyzer as GerSentimentIntensityAnalyzer

def build_sentiment_df(vector_data, lang="en"):
    """
    Erstellt Sentiment-Scores je Dokument.
    Enthält doc_id für sauberen Merge mit Topic-Zuordnung.
    """

    analyzer = SentimentIntensityAnalyzer()

    documents = vector_data["documents"]
    records = []

    for doc_id, doc in enumerate(documents):
        scores = analyzer.polarity_scores(str(doc))
        compound = scores["compound"]

        pos_threshold = 0.5
        neg_threshold = -0.5

        if compound >= pos_threshold:
            label = "positive"
        elif compound <= neg_threshold:
            label = "negative"
        else:
            label = "neutral"

        records.append({
            "doc_id": doc_id,
            "document": doc,
            "sentiment_score": compound,
            "sentiment_label": label,
            "sentiment_pos": scores["pos"],
            "sentiment_neu": scores["neu"],
            "sentiment_neg": scores["neg"]
        })

    return pd.DataFrame(records)

def get_selected_topic_model(selection):
    """
    Extract the relevant components of a selected topic model.

    Parameters
    ----------
    selection : dict
        Dictionary containing:
        - row: selected tuning result (DataFrame row)
        - vector_topic: vectorized dataset

    Returns
    -------
    dict
        Contains model, topic terms, document-topic matrix,
        and related metadata.
    """

    row = selection["row"]
    data = selection["vector_topic"]

    return {
        "model": row["model"],
        "topic_terms": row["topic_terms"],
        "doc_topic_matrix": row["doc_topic_matrix"],
        "documents": data["documents"],
        "features": data["feature_names"],
    }


def compute_dominant_topics(doc_topic_matrix, documents, method):
    """
    Determine the dominant topic for each document.

    For LDA:
        - Uses probability distribution directly
    For LSA:
        - Uses absolute values due to signed components

    Parameters
    ----------
    doc_topic_matrix : np.ndarray
        Document-topic matrix.

    documents : list of str
        Original documents.

    method : str
        Either "lda" or "lsa".

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - document ID
        - document text
        - dominant topic
        - topic strength
    """

    if method == "lda":
        dominant_topics = doc_topic_matrix.argmax(axis=1)
        topic_strengths = doc_topic_matrix.max(axis=1)

    elif method == "lsa":
        abs_matrix = np.abs(doc_topic_matrix)
        dominant_topics = abs_matrix.argmax(axis=1)
        topic_strengths = abs_matrix.max(axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.DataFrame({
        "doc_id": range(len(documents)),
        "document": documents,
        "dominant_topic": dominant_topics + 1,  # 1-based indexing
        "topic_strength": topic_strengths
    })