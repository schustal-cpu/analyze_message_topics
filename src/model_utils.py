# src/model_utils.py

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def build_sentiment_df(model_result, lang, pos=0.05, neg=-0.05):
    """
    Berechnet Sentiment Score + Label pro Dokument.

    Parameter:
        model_result: enthält 'documents'
        lang: 'de' oder 'en'
        pos/neg: Schwellenwerte für Klassifikation

    Rückgabe:
        DataFrame mit:
        - document
        - sentiment_score
        - sentiment_label
    """

    analyzer_en = SentimentIntensityAnalyzer()
    analyzer_de = GerSentimentIntensityAnalyzer()
    
    documents = model_result["documents"]

    # richtigen Analyzer wählen
    if lang == "en":
        analyzer = analyzer_en
    elif lang == "de":
        analyzer = analyzer_de
    else:
        raise ValueError("Unsupported language")

    # Scores berechnen
    scores = [
        analyzer.polarity_scores(doc)["compound"]
        for doc in documents
    ]

    # DataFrame erstellen
    df = pd.DataFrame({
        "document": documents,
        "sentiment_score": scores
    })

    # Label direkt inline
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda s: "positive" if s >= pos else "negative" if s <= neg else "neutral"
    )

    return df


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