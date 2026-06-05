# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (analyze_m_topics)
#     language: python
#     name: analyze-topics-venv
# ---

# +
import os
import numpy as np
import pandas as pd
import scipy

from IPython.display import display
from tqdm.auto import tqdm

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt

from src.styles import *
from src.visualization import *
from src.preprocessing import *
from src.training import *

# +
##########
# Data Import
##########

# Variable Definition
local_dir = "data"
path_reviews = f"{local_dir}/combined_reviews.csv"
stopword_path = "data/stopwords_custom.json"

# Import data to pandas dataframe
df_reviews = pd.read_csv(path_reviews)

df_reviews["language"] = df_reviews["origin"].map({
    "FragdenStaat": "de",
    "YELP": "en",
})

# Remove duplicate Requests
df_reviews = remove_duplicate_texts(df_reviews, text_col="review")


##########
# Stopword Definition
##########

# Import Sentiment Stopwords based on threshold
sia_en = SentimentIntensityAnalyzer()
de_thres = 2
en_thres = 1.5

vader_words_en = load_vader_words(sia_en, en_thres)
gervader_words_de = load_gervader_words("GerVADER/GERVaderLexicon.txt", de_thres)

# Build Basis-Dictionary
dict_stopw_default = {
    "iter_0": {
        "de_nltk_common": stopwords.words("german"),
        "de_vader_topic_only":gervader_words_de,
        "en_nltk_common": stopwords.words("english"),
        "en_vader_topic_only":vader_words_en,
        
    },
    "iter_1": {
        "de_custom_common": [],
        "de_custom_topic_only": [],
        "en_custom_common": [],
        "en_custom_topic_only": []
    }
}


# +
##########
# Data PreProcessing
##########

top_n = 30       # Anzahl anzuzeigender TopWörter
n_process = 1    # Anzahl paralleler Prozesse, Performace-Tunin

while True:
    # Stopwörter JSON laden oder neu erstellen
    if os.path.exists(stopword_path) and os.path.getsize(stopword_path) > 0:
        print("Stopwort-Datei gefunden -> wird eingelesen")
        dict_stopw = load_stopw_from_json(stopword_path)

        if "iter_0" not in dict_stopw:
            dict_stopw["iter_0"] = dict_stopw_default["iter_0"]
            write_stopw_to_json(dict_stopw, stopword_path)

    else:
        print("Stopwort-Datei fehlt oder ist leer -> wird erstellt")
        dict_stopw = dict_stopw_default
        write_stopw_to_json(dict_stopw, stopword_path)

    # Stopwörter zusammenführen
    stopwords_all = build_stopword_lists_from_iterations(dict_stopw)

    stopw_topic_de, stopw_sent_de = stopwords_all["topic_de"], stopwords_all["sentiment_de"]
    stopw_topic_en, stopw_sent_en = stopwords_all["topic_en"], stopwords_all["sentiment_en"]

    # Cleaning Topic/ Sentiment Data
    print("\nCleaning Data - Topic Modeling")
    data_topic = clean_and_tokenize(df_reviews,stopw_de=stopw_topic_de,stopw_en=stopw_topic_en,n_process=n_process)

    print("\nCleaning Data - Sentiment Analysis")
    data_sentiment = clean_and_tokenize(df_reviews,stopw_de=stopw_sent_de,stopw_en=stopw_sent_en,n_process=n_process)

    # Vergleichstabelle anzeigen
    compare_top_tokens_table = compare_top_tokens(
        data_topic,
        data_sentiment,
        top_n
    )
    print()
    display(
        style_compare_top_tokens(
            compare_top_tokens_table,
            caption=f"Vergleich Top {top_n} Token (de/en)"
        )
    )

    current_iter = get_current_iteration_key(dict_stopw)
    
    print(
        f"\nBitte Top Token prüfen und ggf. unter '{current_iter}' in {stopword_path} ergänzen."
    )
    print("1 = JSON ergänzt, nächste Iteration starten")
    print("2 = fertig, weiter mit Topic Modeling")
    
    choice = input("Auswahl: ").strip()
    
    if choice == "1":
        # JSON neu laden (mit deinen Ergänzungen)
        dict_stopw = load_stopw_from_json(stopword_path)
    
        # nächste Iteration vorbereiten
        add_iter_to_json(dict_stopw, stopword_path)
    
    elif choice == "2":
        print("\nFinale Stopwörter übernommen. Weiter mit Topic Modeling.")
        break
    
    else:
        print("\nUngültige Eingabe. Bitte 1 oder 2 wählen.")


# +
############
# Create Vectors with BoW & TF-IDF
############

# Separate Pipelines: Topic Modeling vs. Sentiment
print("Creating Vectors - Topic Modeling")
vectors_topic = {}
vectors_topic["bow_by_lang"] = vectorize(CountVectorizer, data_topic)
vectors_topic["tfidf_by_lang"] = vectorize(TfidfVectorizer, data_topic)

print("\nCreating Vectors - Sentiment Analysis")
vectors_sentiment = {}
vectors_sentiment["bow_by_lang"] = vectorize(CountVectorizer, data_sentiment)
vectors_sentiment["tfidf_by_lang"] = vectorize(TfidfVectorizer, data_sentiment)


# +
############
# LDA/BoW - Model Tuning (en|de)
############

lda_tuning_results_by_lang = {}

for lang, data in vectors_topic["bow_by_lang"].items():
    lda_tuning_results_by_lang[lang] = tune_lda_models(
        data=data,
        topic_grid=[5, 10, 15, 20],
        alpha_grid=[0.01, 0.1],
        eta_grid=[0.01, 0.1],
        max_iter_grid=[10, 20],
        random_states=[1, 2, 3],
        learning_method="batch",
        lang=lang
    )

# +
############
# LDA/BoW - Tuning Validation (en|de)
############

lda_tuning_config = [
    {"col": "coherence",               "weight": 0.30, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.2f}"},
    {"col": "coherence_std",           "weight": 0.10, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.2f}"},
    
    {"col": "perplexity",              "weight": 0.20, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    #{"col": "perplexity_std",          "weight": 0.05, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.2f}"},
    
    {"col": "largest_topic_share",     "weight": 0.15, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    #{"col": "largest_topic_share_std", "weight": 0.05, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.2f}"},
    
    {"col": "avg_word_overlap",        "weight": 0.10, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    #{"col": "max_word_overlap",        "weight": 0.05, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.2f}"}
]

lda_tuning_interpreted_by_lang = {}

for lang, tuning_df_lang in lda_tuning_results_by_lang.items():

    tuning_interpreted = evaluate_and_score_tuning_generic(
        tuning_df=tuning_df_lang,
        score_metrics=lda_tuning_config
    )

    lda_tuning_interpreted_by_lang[lang] = tuning_interpreted

    print(f"\n=== LDA/BoW Tuning ({lang}) ===")

    display(
        style_tuning_eval_generic(
            tuning_interpreted.head(15),  # ← hier begrenzen
            style_metrics=lda_tuning_config,
            hide_cols=["topic_words", "topic_terms", "doc_topic_matrix", "model", "max_word_overlap", "largest_topic_share_std", "perplexity_std"]
        )
    )

# +
############
# LSA/TF-IDF - Model Tuning (en|de)
############

lsa_tuning_results_by_lang = {}

for lang, data in vectors_topic["tfidf_by_lang"].items():
    lsa_tuning_results_by_lang[lang] = tune_lsa_models(
        data=data,

        # Anzahl semantischer Komponenten
        # mehr = feinere Trennung, aber schwerer interpretierbar
        topic_grid=[10, 20, 30, 40],

        # Iterationen der randomized SVD
        # mehr = stabilere Approximation, aber langsamer
        n_iter_grid=[10, 20, 30],

        # fixer Seed für reproduzierbare Ergebnisse
        random_state=42,

        # nur für Progress/Logging
        lang=lang
    )

# +
############
# LSA/TF-IDF - Tuning Validation (en|de)
############

lsa_tuning_config = [
    {"col": "avg_word_overlap",     "weight": 0.16, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    {"col": "coherence",            "weight": 0.42, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "explained_variance",   "weight": 0.17, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "largest_topic_share",  "weight": 0.25, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
]

lsa_tuning_interpreted_by_lang = {}

for lang, tuning_df_lang in lsa_tuning_results_by_lang.items():

    tuning_interpreted = evaluate_and_score_tuning_generic(
        tuning_df=tuning_df_lang,
        score_metrics=lsa_tuning_config
    )

    lsa_tuning_interpreted_by_lang[lang] = tuning_interpreted

    print(f"\n=== LSA/TF-IDF Tuning ({lang}) ===")

    display(
        style_tuning_eval_generic(
            tuning_interpreted.head(10),
            style_metrics=lsa_tuning_config,
            hide_cols=["topic_words", "topic_terms", "doc_topic_matrix", "model"]
        )
    )
# -

# Zeigt die wichtigsten Wörter je Topic
# Erkenntnis:
# - Sind Topics semantisch klar/interpretiertbar?
# - Gibt es generische oder unscharfe Topics?
# - Sind Topics redundant oder klar getrennt?
plot_multiple_topics(best_row)

# Zeigt Überschneidungen der Top-Wörter zwischen Topics
# Erkenntnis:
# - Gibt es redundante/ähnliche Topics?
# - Wurden zu viele Topics erzeugt?
# - Wie gut ist die Topic-Separation?
plot_topic_overlap_matrix(best_row)

# +
############
# LSA/TF-IDF - TopWort Vergleich der Top3 Scores (en|de)
############

display_top_topic_models(
    tuning_interpreted_by_lang=lsa_tuning_interpreted_by_lang,
    model_name="LSA/TF-IDF",
    top_k=3
)

display_top_topic_models(
    tuning_interpreted_by_lang=lda_tuning_interpreted_by_lang,
    model_name="LDA/BoW",
    top_k=3
)

# +
############
# FINAL TOPIC + SENTIMENT PIPELINE
# Separate Vector Spaces:
# - Topic Modeling
# - Sentiment Analysis
############


############
# 2. Manually Select Final Models
############
# IDs = manually evaluated best models

selected_models = {

    "de": {
        "method": "lda",
        "title": "LDA BoW (de)",

        "row": lda_tuning_interpreted_by_lang["de"].iloc[0],

        "vector_topic": vectors_topic["bow_by_lang"]["de"],
        "vector_sentiment": vectors_sentiment["bow_by_lang"]["de"],
    },

    "en": {
        "method": "lsa",
        "title": "LSA TFIDF (en)",

        "row": lsa_tuning_interpreted_by_lang["en"].iloc[3],

        "vector_topic": vectors_topic["tfidf_by_lang"]["en"],
        "vector_sentiment": vectors_sentiment["tfidf_by_lang"]["en"],
    }
}


############
# 3. Extract Final Topic Models
############

def get_selected_topic_model(selection):

    row = selection["row"]
    data = selection["vector_topic"]

    return {
        "model": row["model"],
        "topic_terms": row["topic_terms"],
        "doc_topic_matrix": row["doc_topic_matrix"],
        "documents": data["documents"],
        "features": data["feature_names"],
    }


fitted_models = {}

for lang, selection in selected_models.items():

    fitted_models[lang] = get_selected_topic_model(selection)


############
# 4. Determine Dominant Topic per Document
############

doc_topics = {}

for lang, selection in selected_models.items():

    result = get_selected_topic_model(selection)

    doc_topic_matrix = result["doc_topic_matrix"]
    documents = result["documents"]

    method = selection["method"]

    if method == "lda":
        dominant_topics = doc_topic_matrix.argmax(axis=1)
        topic_strengths = doc_topic_matrix.max(axis=1)

    elif method == "lsa":
        abs_matrix = np.abs(doc_topic_matrix)
        dominant_topics = abs_matrix.argmax(axis=1)
        topic_strengths = abs_matrix.max(axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    doc_topics[lang] = pd.DataFrame({
        "doc_id": range(len(documents)),
        "document": documents,
        "dominant_topic": dominant_topics + 1,
        "topic_strength": topic_strengths
    })

print(doc_topics.keys())

############
# 5. Sentiment Analysis
############

sentiments = {}

for lang, selection in selected_models.items():

    sentiments[lang] = build_sentiment_df(
        selection["vector_sentiment"],
        lang=lang
    )

    
############
# 6. Merge Topic + Sentiment
############

doc_topics_sentiment = {}

for lang in doc_topics.keys():

    doc_topics_sentiment[lang] = doc_topics[lang].merge(
        sentiments[lang],
        on="doc_id",
        how="left",
        suffixes=("_topic", "_sentiment")
    )

############
# 7. Aggregate Topic + Sentiment Matrix
############

topic_sentiment_summary = {}

for lang, df in doc_topics_sentiment.items():

    topic_sentiment_summary[lang] = (
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


############
# 8. Display Final Topic + Sentiment Matrices
############

for lang, summary_df in topic_sentiment_summary.items():

    selection = selected_models[lang]

    print(f"\n=== Topic + Sentiment Vergleich ({lang}) ===")

    display(
        style_topic_sentiment(
            summary_df,
            f"{selection['title']} - Topic & Sentiment Matrix",
            pos_threshold=0.1,
            neg_threshold=-0.1
        )
    )


# +
#display_selected_model_topic_terms(
#    selected_models,
#    weight_col="weight_pct"
#)

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
#doc_topics_sentiment["en"].head()
#doc_topics_sentiment["de"].head()
pd.set_option("display.max_colwidth", 1000)
show_reviews_for_topic(
    doc_topics_sentiment,
    lang="en",
    topic_id=1,
    sentiment_label="negative",
    n=10
)
# -

# Zeigt die Anzahl Dokumente pro dominantem Topic
# Erkenntnis:
# - Sind Topics ausgewogen verteilt?
# - Dominiert ein einzelnes Topic zu stark?
# - Gibt es irrelevante Mini-Topics?
plot_topic_sizes(doc_topics["en"])
plot_topic_sizes(doc_topics["de"])

# +
# Zeigt durchschnittliches Sentiment pro Topic
# Erkenntnis:
# - Welche Themen sind positiv/negativ?
# - Gibt es emotional stark geladene Topics?
# - Wie unterscheiden sich Topics semantisch/emotional?
plot_topic_sentiment(
    topic_sentiment_summary["en"],
    title="EN Topic Sentiment"
)

plot_topic_sentiment(
    topic_sentiment_summary["de"],
    title="DE Topic Sentiment"
)


# +

def display_selected_model_topic_terms(selected_models, weight_col="weight_pct"):
    """
    Gibt die Top-Wörter pro Topic für manuell ausgewählte Modelle aus.
    Funktioniert mit Struktur:
    selected_models["de"]["row"]
    selected_models["en"]["row"]
    """

    for lang, selection in selected_models.items():

        row = selection["row"]
        title = selection["title"]

        score = row.get("overall_score", None)
        n_topics = row.get("n_topics", len(row["topic_terms"]))

        print(f"\n=== Ausgewähltes Topic-Modell ({lang}) ===")

        if score is not None:
            print(
                f"\n--- {title} | "
                f"Score: {score:.4f} | "
                f"Topics: {n_topics} ---"
            )
        else:
            print(
                f"\n--- {title} | "
                f"Topics: {n_topics} ---"
            )

        display(topic_terms_to_table(row, weight_col=weight_col))


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


# +
def plot_topic_sentiment_from_summary(summary, method_key, lang):
    df = summary[method_key][lang]

    plot_df = df.set_index("dominant_topic")[[
        "positive", "neutral", "negative"
    ]]

    plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5)
    )

    plt.title(f'{methods[method_key]["title"]} ({lang}) - Topic + Sentiment')
    plt.xlabel("Topic")
    plt.ylabel("Anzahl Dokumente")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_topic_sentiment_from_summary(topic_sentiment_summary, "bow_lda", "de")


# +
def topic_terms_to_table(row, weight_col="weight_pct"):
    return pd.DataFrame({
        topic: [
            f'{item["word"]} ({item[weight_col]:.1f}%)'
            for item in terms
        ]
        for topic, terms in row["topic_terms"].items()
    })

    return pd.DataFrame(records)
def display_top_topic_models(tuning_interpreted_by_lang, model_name, top_k=3, weight_col="weight_pct"):
    for lang, tuning_df in tuning_interpreted_by_lang.items():

        print(f"\n=== Top {top_k} {model_name} Modelle ({lang}) ===")

        for rank, (idx, row) in enumerate(tuning_df.head(top_k).iterrows(), start=1):
            print(
                f"\n--- Rang {rank} | Index {idx} | "
                f"Score: {row['overall_score']:.4f} | "
                f"Topics: {row['n_topics']} ---"
            )

            display(topic_terms_to_table(row, weight_col=weight_col))


def plot_topic_words(row, topic_id=1, top_n=10, title=None):
    topic_key = f"Topic {topic_id}"
    terms = row["topic_terms"][topic_key][:top_n]

    words = [t["word"] for t in terms][::-1]
    weights = [t["abs_weight"] for t in terms][::-1]

    plt.figure(figsize=(8, 4))
    plt.barh(words, weights)
    plt.xlabel("Weight")
    plt.title(title or f"{topic_key} - Top Words")
    plt.tight_layout()
    plt.show()
    
def plot_multiple_topics(row, topic_ids=None, top_n=10):
    topic_terms = row["topic_terms"]

    if topic_ids is None:
        topic_ids = range(1, len(topic_terms) + 1)

    for topic_id in topic_ids:
        plot_topic_words(
            row=row,
            topic_id=topic_id,
            top_n=top_n,
            title=f"Topic {topic_id}"
        )

def plot_multiple_topics(row, topic_ids=None, top_n=10):
    topic_terms = row["topic_terms"]

    if topic_ids is None:
        topic_ids = range(1, len(topic_terms) + 1)

    for topic_id in topic_ids:
        plot_topic_words(
            row=row,
            topic_id=topic_id,
            top_n=top_n,
            title=f"Topic {topic_id}"
        )

def plot_topic_term_heatmap(row, top_n=10, title="Topic-Term Heatmap"):
    topic_terms = row["topic_terms"]

    records = []

    for topic, terms in topic_terms.items():
        for term in terms[:top_n]:
            records.append({
                "topic": topic,
                "word": term["word"],
                "weight": term["abs_weight"]
            })

    df = pd.DataFrame(records)

    heatmap_data = (
        df.pivot_table(
            index="word",
            columns="topic",
            values="weight",
            fill_value=0
        )
    )

    plt.figure(figsize=(12, max(5, len(heatmap_data) * 0.25)))
    plt.imshow(heatmap_data, aspect="auto")
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=90)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.tight_layout()
    plt.show()

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

def plot_topic_overlap_matrix(row, title="Topic Word Overlap Matrix"):
    overlap_df = compute_topic_overlap_matrix(row)

    plt.figure(figsize=(8, 6))
    plt.imshow(overlap_df, aspect="auto")
    plt.xticks(range(len(overlap_df.columns)), overlap_df.columns, rotation=90)
    plt.yticks(range(len(overlap_df.index)), overlap_df.index)
    plt.colorbar(label="Jaccard Overlap")
    plt.title(title)
    plt.tight_layout()
    plt.show()
def plot_topic_sizes(doc_topic_df, title="Topic Sizes"):
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

