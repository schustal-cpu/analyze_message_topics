# src/evaluation.py

import pandas as pd


def merge_topic_sentiment(topic_df, sentiment_df):
    """
    Merge topic assignments with sentiment results.

    Parameters
    ----------
    topic_df : pd.DataFrame
        Contains document topics.

    sentiment_df : pd.DataFrame
        Contains sentiment scores and labels.

    Returns
    -------
    pd.DataFrame
        Combined dataset with topic + sentiment information.
    """

    return topic_df.merge(
        sentiment_df,
        on="doc_id",
        how="left",
        suffixes=("_topic", "_sentiment")
    )


def build_topic_sentiment_summary(df):
    """
    Aggregate topic and sentiment data.

    Computes:
    - number of documents per topic
    - average topic strength
    - average sentiment score
    - sentiment distribution (positive / neutral / negative)

    Parameters
    ----------
    df : pd.DataFrame
        Combined topic + sentiment DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated summary per topic.
    """

    return (
        df.groupby("dominant_topic")
        .agg(
            documents=("document_topic", "count"),
            avg_topic_strength=("topic_strength", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            positive=("sentiment_label", lambda x: (x == "positive").sum()),
            neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
            negative=("sentiment_label", lambda x: (x == "negative").sum())
        )
        .reset_index()
    )