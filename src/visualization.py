#### Analyse Funktionen #######

from collections import Counter
import pandas as pd
from src.styles import *

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

from collections import Counter
import pandas as pd
from IPython.display import display


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

    # Optional display (keeps function reusable!)
    if show:
        display(
            style_compare_top_tokens(
                df,
                caption=f"Top {top_n} Token Comparison (de/en)"
            )
        )

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