#### Analyse Funktionen #######

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from src.styles import (
    style_tuning_eval_generic,
    style_compare_top_tokens,
)

def evaluate_and_score_tuning_generic(
    tuning_df,
    score_metrics,
    score_col="overall_score"
):
    df = tuning_df.copy()
    score = 0

    for metric in score_metrics:
        col = metric["col"]
        weight = metric["weight"]
        higher_is_better = metric["higher_is_better"]

        min_val = df[col].min()
        max_val = df[col].max()

        if max_val == min_val:
            norm = 1
        else:
            norm = (df[col] - min_val) / (max_val - min_val)

        if not higher_is_better:
            norm = 1 - norm

        score += weight * norm

    df[score_col] = score

    return df.sort_values(score_col, ascending=False)
    
def display_tuning_results(results_by_lang, tuning_config, title, head_rows, hide_cols):
    """
    Evaluate, score, and display tuning results for multiple languages.

    For each language-specific tuning DataFrame, this function:
    1. Applies a generic scoring and evaluation based on the provided metrics
    2. Stores the interpreted results in a dictionary
    3. Prints a formatted section header
    4. Displays a styled preview of the top results

    Parameters
    ----------
    results_by_lang : dict
        Dictionary mapping language codes (e.g. 'en', 'de') to tuning result DataFrames.

    tuning_config : list of dict
        Configuration describing evaluation metrics.
        Each entry defines:
            - column name
            - weight
            - optimization goal (higher/lower is better)
            - quantiles for good/bad thresholds
            - display format

    title : str
        Title used in the printed section header (e.g. "LDA/BoW Tuning").

    head_rows : int
        Number of top rows to display per language after scoring.

    hide_cols : list of str
        Columns to hide in the styled output display.

    Returns
    -------
    dict
        Dictionary with the same language keys as input, containing
        evaluated and scored tuning DataFrames.
    """

    # Dictionary to store evaluated results per language
    interpreted_by_lang = {}

    # Iterate over each language and its corresponding tuning DataFrame
    for lang, tuning_df_lang in results_by_lang.items():

        # Apply scoring and evaluation logic using the provided configuration
        tuning_interpreted = evaluate_and_score_tuning_generic(
            tuning_df=tuning_df_lang,
            score_metrics=tuning_config
        )

        # Store the evaluated results
        interpreted_by_lang[lang] = tuning_interpreted

        # Print section header for better readability in output
        print(f"\n=== {title} ({lang}) ===")

        # Display a styled preview of the top tuning results
        display(
            style_tuning_eval_generic(
                tuning_interpreted.head(head_rows),  # limit number of displayed rows
                style_metrics=tuning_config,
                hide_cols=hide_cols
            )
        )

    # Return all evaluated tuning results grouped by language
    return interpreted_by_lang

def compare_top_tokens(data_topic, data_sentiment, top_n=30, show=True):
    """
    Compare the most frequent tokens for topic modeling and sentiment analysis
    across German and English datasets.

    The function computes token frequencies per language and task, builds a
    structured comparison table, and optionally renders a styled display.

    Parameters
    ----------
    data_topic : dict
        Preprocessed topic modeling data (output from clean_and_tokenize).

    data_sentiment : dict
        Preprocessed sentiment analysis data.

    top_n : int, optional (default=30)
        Number of top tokens to compare.

    show : bool, optional (default=True)
        If True, the result is displayed using the styling function.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame comparing tokens and counts for:
        - German vs English
        - Topic vs Sentiment
    """

    def get_counts(data, lang):
        """Extract most common tokens for a given language."""
        tokens = [
            word
            for sent in data[lang]["sentences"]
            for word in sent
        ]
        return Counter(tokens).most_common(top_n)

    # Compute token frequencies
    topic_de = get_counts(data_topic, "de")
    sent_de  = get_counts(data_sentiment, "de")

    topic_en = get_counts(data_topic, "en")
    sent_en  = get_counts(data_sentiment, "en")

    # Build comparison DataFrame
    df = pd.DataFrame({
        ("de", "Topic", "Token"): pd.Series([w for w, _ in topic_de]),
        ("de", "Topic", "Count"): pd.Series([c for _, c in topic_de]),
    
        ("de", "Sentiment", "Token"): pd.Series([w for w, _ in sent_de]),
        ("de", "Sentiment", "Count"): pd.Series([c for _, c in sent_de]),
    
        ("en", "Topic", "Token"): pd.Series([w for w, _ in topic_en]),
        ("en", "Topic", "Count"): pd.Series([c for _, c in topic_en]),
    
        ("en", "Sentiment", "Token"): pd.Series([w for w, _ in sent_en]),
        ("en", "Sentiment", "Count"): pd.Series([c for _, c in sent_en]),
    })
    
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # Optional display (keeps function reusable!)
    if show:
        display(
            style_compare_top_tokens(
                df,
                caption=f"Top {top_n} Token Comparison (de/en)"
            )
        )

    return df

def print_tuning_legend():
    print("\n=== Legende zur Parameterbewertung ===\n")

    print("COHERENCE:")
    print("  ↑ höher = besser")
    print("  → misst, wie gut Wörter innerhalb eines Topics zusammenpassen")
    print("  → gute Topics: thematisch konsistente Wörter\n")

    print("PERPLEXITY:")
    print("  ↓ niedriger = besser (nur für LDA relevant)")
    print("  → misst, wie gut das Modell die Daten erklärt")
    print("  → Achtung: nicht allein zur Bewertung verwenden!\n")

    print("LARGEST_TOPIC_SHARE:")
    print("  ↓ niedriger = besser")
    print("  → Anteil des größten Topics")
    print("  → gut: < 0.35 (ausgewogene Topics)")
    print("  → schlecht: > 0.50 (ein Topic dominiert)\n")

    print("N_TOPICS:")
    print("  → Anzahl Topics")
    print("  → zu klein: Themen vermischen sich")
    print("  → zu groß: Topics werden redundant oder leer\n")

    print("ALPHA (doc_topic_prior):")
    print("  ↓ kleiner = besser interpretierbar")
    print("  → steuert, wie viele Topics ein Dokument hat")
    print("  → klein (0.01–0.1): klare Topic-Zuordnung")
    print("  → groß: Dokumente enthalten viele Topics (unscharf)\n")

    print("ETA (topic_word_prior):")
    print("  ↓ kleiner = bessere Topics")
    print("  → steuert, wie viele Wörter ein Topic dominieren")
    print("  → klein (0.01–0.1): klare, prägnante Topics")
    print("  → groß: viele generische Wörter im Topic\n")

    print("GESAMTBEWERTUNG (overall_eval):")
    print("  good = gute Parameterkombination")
    print("  ok   = brauchbar, aber nicht optimal")
    print("  bad  = vermeiden\n")

    print("EMPFEHLUNG:")
    print("  → Wähle Modelle mit:")
    print("     - hoher coherence")
    print("     - niedriger largest_topic_share")
    print("     - stabilen, interpretierbaren Topics\n")

def print_overall_score_info():
    print("\n=== Erklärung Overall Score ===\n")

    print("Der Overall Score bewertet die Qualität eines Topic-Modells (0 bis 1).")
    print("Je höher der Score, desto besser das Modell.\n")

    print("Er setzt sich aus drei Komponenten zusammen:\n")

    print("1. Coherence (Gewichtung: 50%)")
    print("   → misst, wie gut Wörter innerhalb eines Topics zusammenpassen")
    print("   → höher = besser\n")

    print("2. Topic Balance (Gewichtung: 30%)")
    print("   → misst, wie gleichmäßig die Dokumente auf Topics verteilt sind")
    print("   → niedriger Anteil des größten Topics = besser\n")

    print("3. Perplexity (Gewichtung: 20%)")
    print("   → misst, wie gut das Modell die Daten erklärt")
    print("   → niedriger = besser (wird intern invertiert)\n")

    print("Alle Werte werden normalisiert und kombiniert:\n")

    print("   Overall Score = 0.5 * Coherence")
    print("                 + 0.3 * Topic Balance")
    print("                 + 0.2 * Perplexity\n")

    print("Interpretation:")
    print("   > 0.8   → sehr gut")
    print("   0.6–0.8 → gut")
    print("   0.4–0.6 → ok")
    print("   < 0.4   → schwach\n")

    print("Hinweis:")
    print("   Der Score ist eine Orientierungshilfe.")
    print("   Die inhaltliche Interpretation der Topics bleibt entscheidend.\n")



def topic_terms_to_table(row, weight_col="weight_pct"):
    return pd.DataFrame({
        topic: [
            f'{item["word"]} ({item[weight_col]:.1f}%)'
            for item in terms
        ]
        for topic, terms in row["topic_terms"].items()
    })

    return pd.DataFrame(records)
def display_topic_model_summary(tuning_interpreted_by_lang, model_name, top_k=3, weight_col="weight_pct"):
    for lang, tuning_df in tuning_interpreted_by_lang.items():

        print(f"\n=== Top {top_k} {model_name} Modelle ({lang}) ===")

        for rank, (idx, row) in enumerate(tuning_df.head(top_k).iterrows(), start=1):
            print(
                f"\n--- Rang {rank} | Index {idx} | "
                f"Score: {row['overall_score']:.4f} | "
                f"Topics: {row['n_topics']} ---"
            )

            display(topic_terms_to_table(row, weight_col=weight_col))



def plot_topic_words(data, topic_ids=None, top_n=10):
    if isinstance(data, pd.Series):
        data = data.to_frame().T

    for model_idx, row in data.iterrows():

        topic_terms = row["topic_terms"]
        n_topics = row.get("n_topics", len(topic_terms))
        score = row.get("score", row.get("overall_score", None))

        score_text = (
            f" | Score: {score:.4f}"
            if score is not None
            else ""
        )

        print(
            f"\n=== Model idx {model_idx} | "
            f"{n_topics} Topics"
            f"{score_text} ==="
        )

        if topic_ids is None:
            topics = range(1, len(topic_terms) + 1)
        else:
            topics = topic_ids

        for topic_id in topics:
            topic_key = f"Topic {topic_id}"

            if topic_key not in topic_terms:
                print(f"Skipping {topic_key}: not found.")
                continue

            terms = topic_terms[topic_key][:top_n]

            words = [t["word"] for t in terms][::-1]
            weights = [t["abs_weight"] for t in terms][::-1]

            plt.figure(figsize=(8, 4))
            plt.barh(words, weights)
            plt.xlabel("Weight")
            plt.title(f"Model idx {model_idx} - {topic_key}")
            plt.tight_layout()
            plt.show()
        
def compare_topic_sizes(results_df, method="lsa", title=None):

    if title is not None:
        print(title)

    tables = []

    for idx, row in results_df.iterrows():
        matrix = row["doc_topic_matrix"]

        dominant = (
            matrix.argmax(axis=1)
            if method == "lda"
            else np.abs(matrix).argmax(axis=1)
        )

        counts = (
            pd.Series(dominant)
            .value_counts()
            .sort_index()
            .rename(f"Docs (Model {idx})")
        )

        tables.append(counts)

    return (
        pd.concat(tables, axis=1)
        .fillna(0)
        .astype(int)
        .rename_axis("Topic")
        .reset_index()
    )
    

def compute_topic_overlap_matrix(row):
    topics = {
        topic: [item["word"] for item in terms]
        for topic, terms in row["topic_terms"].items()
    }

    topic_names = list(topics.keys())
    matrix = np.zeros((len(topic_names), len(topic_names)))

    for i, topic_a in enumerate(topic_names):
        for j, topic_b in enumerate(topic_names):
            set_a = set(topics[topic_a])
            set_b = set(topics[topic_b])

            union = set_a | set_b
            intersection = set_a & set_b

            matrix[i, j] = len(intersection) / len(union) if union else 0

    return pd.DataFrame(matrix, index=topic_names, columns=topic_names)

def plot_topic_overlap_matrix(data, title="Topic Word Overlap Matrix"):
    """
    Plots the topic overlap heatmap for one or multiple models.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Single model (row) or multiple models.
    title : str
        Base title for the plot(s).
    """

    # Einzelnes Modell
    if isinstance(data, pd.Series):
        data = data.to_frame().T

    # Mehrere Modelle
    for idx, row in data.iterrows():
        overlap_df = compute_topic_overlap_matrix(row)

        plt.figure(figsize=(8, 6))
        plt.imshow(overlap_df, aspect="auto")
        plt.xticks(range(len(overlap_df.columns)), overlap_df.columns, rotation=90)
        plt.yticks(range(len(overlap_df.index)), overlap_df.index)
        plt.colorbar(label="Jaccard Overlap")

        model_title = (
            f"{title} (idx {idx}, {row['n_topics']} Topics)"
            if "n_topics" in row
            else f"{title} (idx {idx})"
        )

        plt.title(model_title)
        plt.tight_layout()
        plt.show()


def plot_topic_sizes(
    doc_topic_df=None,
    doc_topic_matrix=None,
    method="lda",
    title="Topic Sizes"
):
    if doc_topic_df is None:

        if method == "lda":
            dominant = doc_topic_matrix.argmax(axis=1)
        elif method == "lsa":
            dominant = np.abs(doc_topic_matrix).argmax(axis=1)
        else:
            raise ValueError("method must be 'lda' or 'lsa'")

        counts = (
            pd.Series(dominant)
            .value_counts()
            .sort_index()
        )

    else:
        counts = (
            doc_topic_df["dominant_topic"]
            .value_counts()
            .sort_index()
        )

    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Topic")
    plt.ylabel("Documents")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def plot_topic_sentiment(df, title="Topic Sentiment"):

    ax = df.plot(
        x="dominant_topic",
        y="avg_sentiment",
        kind="bar",
        legend=False,
        figsize=(12, 5)
    )

    ax.set_title(title)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Average Sentiment")

    plt.axhline(0, color="black", linewidth=1)

    plt.show()

def show_reviews_for_topic(
    doc_topics_sentiment,
    lang,
    topic_id,
    sentiment_label=None,
    n=10,
    sort_by="topic_strength"
):
    df = doc_topics_sentiment[lang].copy()

    df = df[df["dominant_topic"] == topic_id]

    if sentiment_label is not None:
        df = df[df["sentiment_label"] == sentiment_label]

    df = df.sort_values(sort_by, ascending=False)

    cols = [
        "doc_id",
        "dominant_topic",
        "topic_strength",
        "sentiment_score",
        "sentiment_label"
    ]

    # Falls du suffixes beim Merge hast
    if "document_topic" in df.columns:
        cols.append("document_topic")
    elif "document" in df.columns:
        cols.append("document")

    return df[cols].head(n)
    