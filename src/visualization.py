#### Analyse Funktionen #######

from collections import Counter
import pandas as pd

def analyze_tokens(data_by_lang, top_n=30):
    data = {}

    for lang in ["de", "en"]:
        sentences = data_by_lang[lang]["sentences"]
        tokens = [word for sent in sentences for word in sent]
        counts = Counter(tokens).most_common(top_n)

        data[(lang, "Token")] = [w for w, _ in counts]
        data[(lang, "Count")] = [c for _, c in counts]

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return style_token_table(
        df,
        split_after_col=1,
        count_cols=[("de", "Count"), ("en", "Count")],
        caption=f"Top {top_n} Tokens (de vs en)"
    )

def compare_top_tokens(data_topic, data_sentiment, top_n=30):
    """
    Vergleich Topic vs Sentiment für DE und EN gleichzeitig
    """

    def get_counts(data, lang):
        tokens = [
            word
            for sent in data[lang]["sentences"]
            for word in sent
        ]
        return Counter(tokens).most_common(top_n)

    # Counts holen
    topic_de = get_counts(data_topic, "de")
    sent_de  = get_counts(data_sentiment, "de")

    topic_en = get_counts(data_topic, "en")
    sent_en  = get_counts(data_sentiment, "en")

    # DataFrame bauen
    df = pd.DataFrame({
        ("de", "Topic", "Token"): [w for w, _ in topic_de],
        ("de", "Topic", "Count"): [c for _, c in topic_de],
        ("de", "Sentiment", "Token"): [w for w, _ in sent_de],
        ("de", "Sentiment", "Count"): [c for _, c in sent_de],

        ("en", "Topic", "Token"): [w for w, _ in topic_en],
        ("en", "Topic", "Count"): [c for _, c in topic_en],
        ("en", "Sentiment", "Token"): [w for w, _ in sent_en],
        ("en", "Sentiment", "Count"): [c for _, c in sent_en],
    })

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df
    
def print_lda_topics(model, feature_names, n_top_words=10):
    """
    Gibt die wichtigsten Wörter je Topic aus.
    """
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]

        print(f"\nTopic {topic_idx + 1}:")
        print(", ".join(top_words))

def topics_matrix(model, feature_names, n_top_words=10, prefix="Topic"):
    """
    Erstellt eine Topic-Tabelle für ein einzelnes Modell.
    Spalten: Topic1, Topic2, ...
    Zeilen: Top-Wörter
    """
    topics = {}

    for i, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-n_top_words:][::-1]
        topics[f"{prefix}{i+1}"] = [feature_names[j] for j in top_idx]

    return pd.DataFrame(topics)


def compare_topic_models_multiindex(models, n_top_words=10):
    first_model = next(iter(models.values()))["model"]
    n_topics = first_model.components_.shape[0]

    data = {}

    for topic_idx in range(n_topics):
        for method_name, d in models.items():
            topic = d["model"].components_[topic_idx]
            features = d["features"]

            top_idx = topic.argsort()[-n_top_words:][::-1]
            words = [features[i] for i in top_idx]

            data[(f"Topic {topic_idx+1}", method_name)] = words

    return pd.DataFrame(data)

def build_topic_comparison_tables(doc_topics, methods, lang):
    
    def to_word_list(words):
        if isinstance(words, list):
            return words
        return [w.strip() for w in str(words).split(",")]

    rows = []

    for method_key, lang_results in doc_topics.items():
        df = lang_results[lang]
        title = methods[method_key]["title"]

        # Topic → Keywords (ein Eintrag pro Topic)
        topic_map = (
            df.groupby("dominant_topic")["topic_keywords"]
            .first()
            .sort_index()
            .apply(to_word_list)
        )

        max_len = max(len(words) for words in topic_map)

        for i in range(max_len):
            row = {"Method": title if i == 0 else ""}

            for topic_id, words in topic_map.items():
                row[f"Topic {topic_id + 1}"] = words[i] if i < len(words) else ""

            rows.append(row)

    return pd.DataFrame(rows)