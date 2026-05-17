# src/analyzer.py

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