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
        topic_grid=[15, 20],                 # Anzahl Topics (zu klein = zu grob, zu groß = redundant)
        alpha_grid=[0.01, 0.05, 0.1, None],  # Dokument→Topic-Verteilung (klein = wenige dominante Topics, groß = mehrere Topics)
        eta_grid=[0.01, 0.05, 0.1, None],    # Topic→Wort-Verteilung (klein = wenige dominante Wörter, groß = breitere Topics)
        max_iter=10,                         # Trainingsdurchläufe (mehr = bessere Konvergenz, langsamer)
        learning_method="batch",             # batch = stabil, online = schneller bei großen Daten
        lang=lang                            # nur für Logging/Progress
    )

# +
############
# LDA/BoW - Tuning Validation (en|de)
############

lda_tuning_config = [
    {"col": "coherence", "weight": 0.5, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "perplexity", "weight": 0.2, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.2f}"},
    {"col": "largest_topic_share", "weight": 0.3, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
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
            hide_cols=["topic_words", "topic_terms", "doc_topic_matrix"]
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
        topic_grid=[15, 20, 25, 30],   # Anzahl Topics / Dimensionen (mehr = feinere Struktur, aber schwerer interpretierbar)
        n_iter_grid=[5, 10, 20],       # Iterationen für SVD (mehr = stabilere Komponenten, aber langsamer)
        lang=lang                      # nur für Logging/Progress
    )

# +
############
# LSA/TF-IDF - Tuning Validation (en|de)
############

lsa_tuning_config = [
    {"col": "coherence", "weight": 0.5, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "explained_variance", "weight": 0.2, "higher_is_better": True, "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "largest_topic_share", "weight": 0.3, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
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
            tuning_interpreted,
            style_metrics=lsa_tuning_config,
            hide_cols=["topic_words", "topic_terms", "doc_topic_matrix"]
        )
    )


# +
def plot_topic_terms(row, topic_id, weight_col="weight_pct", top_n=10):
    topic_key = f"Topic {topic_id}"
    terms = row["topic_terms"][topic_key][:top_n]

    words = [item["word"] for item in terms][::-1]
    values = [item[weight_col] for item in terms][::-1]

    plt.figure(figsize=(8, 4))
    plt.barh(words, values)
    plt.title(f"{topic_key} – Top Wörter")
    plt.xlabel("Gewichtung (%)")
    plt.tight_layout()
    plt.show()
    
def plot_multiple_topics(row, topic_ids, weight_col="weight_pct", top_n=8):
    plt.figure(figsize=(10, 5))

    for topic_id in topic_ids:
        topic_key = f"Topic {topic_id}"
        terms = row["topic_terms"][topic_key][:top_n]

        words = [item["word"] for item in terms]
        values = [item[weight_col] for item in terms]

        plt.plot(values, marker="o", label=topic_key)

    plt.title("Topic Vergleich")
    plt.xlabel("Top-Wörter Rang")
    plt.ylabel("Gewichtung (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_topic_distribution(doc_topic_matrix, method_type="lsa"):
    
    if method_type == "lsa":
        doc_topic_matrix = np.abs(doc_topic_matrix)

    dominant_topics = doc_topic_matrix.argmax(axis=1)

    counts = (
        pd.Series(dominant_topics)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("Topic Verteilung")
    plt.xlabel("Topic")
    plt.ylabel("Anzahl Dokumente")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
best_row = lsa_tuning_interpreted_by_lang["de"].iloc[0]

plot_topic_terms(best_row, topic_id=1)
plot_topic_terms(best_row, topic_id=2)

plot_multiple_topics(best_row, topic_ids=[1, 2, 3])


# +
############
# LSA/TF-IDF - TopWort Vergleich der Top3 Scores (en|de)
############

def topic_terms_to_table(row, weight_col="weight_pct"):
    return pd.DataFrame({
        topic: [
            f'{item["word"]} ({item[weight_col]:.1f}%)'
            for item in terms
        ]
        for topic, terms in row["topic_terms"].items()
    })


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
# Topic + Sentiment Analyse
############

# 1. Manuelle Auswahl finaler Modelle pro Sprache und Methode
selected_models = {
    "de": {
        "lda": {
            "method_type": "lda",
            "title": "LDA BoW",
            "row": lda_tuning_interpreted_by_lang["de"].iloc[0],
            "vector_data": vectors_topic["bow_by_lang"]["de"],
        },
        "lsa": {
            "method_type": "lsa",
            "title": "LSA TF-IDF",
            "row": lsa_tuning_interpreted_by_lang["de"].loc[8],
            "vector_data": vectors_topic["tfidf_by_lang"]["de"],
        },
    },
    "en": {
        "lda": {
            "method_type": "lda",
            "title": "LDA BoW",
            "row": lda_tuning_interpreted_by_lang["en"].iloc[0],
            "vector_data": vectors_topic["bow_by_lang"]["en"],
        },
        "lsa": {
            "method_type": "lsa",
            "title": "LSA TF-IDF",
            "row": lsa_tuning_interpreted_by_lang["en"].iloc[0],
            "vector_data": vectors_topic["tfidf_by_lang"]["en"],
        },
    },
}


# 2. Finales Modell anhand gewählter Tuning-Zeile neu trainieren
def fit_selected_topic_model(selection, random_state=42):
    row = selection["row"]
    method_type = selection["method_type"]
    data = selection["vector_data"]

    X = data["matrix"]

    if method_type == "lda":
        model = LatentDirichletAllocation(
            n_components=int(row["n_topics"]),
            random_state=random_state,
            learning_method="batch",
            max_iter=20,
            doc_topic_prior=None if pd.isna(row["alpha"]) else float(row["alpha"]),
            topic_word_prior=None if pd.isna(row["eta"]) else float(row["eta"]),
            evaluate_every=-1
        )
        doc_topic_matrix = model.fit_transform(X)

    elif method_type == "lsa":
        model = TruncatedSVD(
            n_components=int(row["n_topics"]),
            n_iter=int(row["n_iter"]),
            random_state=random_state
        )
        doc_topic_matrix = model.fit_transform(X)

    else:
        raise ValueError("method_type muss 'lda' oder 'lsa' sein")

    return {
        "model": model,
        "doc_topic_matrix": doc_topic_matrix,
        "documents": data["documents"],
        "features": data["feature_names"],
    }


# 3. Topic-Zuordnung je Dokument berechnen
doc_topics = {}
fitted_models = {}

for lang, methods_dict in selected_models.items():
    doc_topics[lang] = {}
    fitted_models[lang] = {}

    for method_key, selection in methods_dict.items():

        result = fit_selected_topic_model(selection)
        fitted_models[lang][method_key] = result

        doc_topic_matrix = result["doc_topic_matrix"]
        documents = result["documents"]
        method_type = selection["method_type"]

        if method_type == "lda":
            dominant_topics = doc_topic_matrix.argmax(axis=1)
            topic_strengths = doc_topic_matrix.max(axis=1)

        elif method_type == "lsa":
            abs_matrix = np.abs(doc_topic_matrix)
            dominant_topics = abs_matrix.argmax(axis=1)
            topic_strengths = abs_matrix.max(axis=1)

        doc_topics[lang][method_key] = pd.DataFrame({
            "document": documents,
            "dominant_topic": dominant_topics + 1,
            "topic_strength": topic_strengths
        })


# 4. Sentiment je Sprache berechnen
sentiments = {
    lang: build_sentiment_df(result, lang=lang)
    for lang, result in vectors_topic["tfidf_by_lang"].items()
}


# 5. Topic-Zuordnung + Sentiment verbinden
doc_topics_sentiment = {}

for lang, methods_dict in doc_topics.items():
    doc_topics_sentiment[lang] = {}

    for method_key, df_topics in methods_dict.items():
        doc_topics_sentiment[lang][method_key] = df_topics.merge(
            sentiments[lang],
            on="document",
            how="left"
        )


# 6. Aggregierte Topic/Sentiment-Tabelle je Sprache und Modell
topic_sentiment_summary = {}

for lang, methods_dict in doc_topics_sentiment.items():
    topic_sentiment_summary[lang] = {}

    for method_key, df in methods_dict.items():
        topic_sentiment_summary[lang][method_key] = (
            df.groupby("dominant_topic")
            .agg(
                documents=("document", "count"),
                avg_topic_strength=("topic_strength", "mean"),
                avg_sentiment=("sentiment_score", "mean"),
                positive=("sentiment_label", lambda x: (x == "positive").sum()),
                neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
                negative=("sentiment_label", lambda x: (x == "negative").sum())
            )
            .reset_index()
        )


# 7. Ausgabe: LDA vs. LSA je Sprache
for lang, methods_dict in topic_sentiment_summary.items():

    print(f"\n=== Topic + Sentiment Vergleich ({lang}) ===")

    pos_threshold, neg_threshold = (0.5, -0.5) if lang == "de" else (0.1, -0.1)

    for method_key, summary_df in methods_dict.items():
        selection = selected_models[lang][method_key]

        display(
            style_topic_sentiment(
                summary_df,
                f"{selection['title']} ({lang}) - Topic & Sentiment Matrix",
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold
            )
        )

# +
############
# Topic + Sentiment Analyse
############

# 1. Topic-Zuordnung je Modell
doc_topics = {}

for method_key, config in methods.items():
    doc_topics[method_key] = {}

    for lang, result in config["topics_by_lang"].items():

        doc_topic_matrix = result["doc_topic_matrix"]
        documents = result["documents"]
        method_type = config["method_type"]

        # --------------------
        # Dominantes Topic je Dokument
        # --------------------
        if method_type == "lda":
            dominant_topics = doc_topic_matrix.argmax(axis=1)
            topic_strengths = doc_topic_matrix.max(axis=1)

        elif method_type == "lsa":
            dominant_topics = np.abs(doc_topic_matrix).argmax(axis=1)
            topic_strengths = np.abs(doc_topic_matrix).max(axis=1)

        # --------------------
        # DataFrame je Dokument
        # --------------------
        df_topics = pd.DataFrame({
            "document": documents,
            "dominant_topic": dominant_topics + 1,
            "topic_strength": topic_strengths
        })

        doc_topics[method_key][lang] = df_topics


# 2. Sentiment je Sprache berechnen
# Sentiment wird nur anhand TFIDF Vektor berechnet
sentiments = {
    lang: build_sentiment_df(result, lang=lang)
    for lang, result in vectors_topic["tfidf_by_lang"].items()
}

# 3. Topic-Zuordnung + Sentiment verbinden
doc_topics_sentiment = {
    method: {
        lang: df_topics.merge(
            sentiments[lang],
            on="document",
            how="left"
        )
        for lang, df_topics in lang_results.items()
    }
    for method, lang_results in doc_topics.items()
}

# 4. Aggregierte Topic/Sentiment-Tabelle je Modell
topic_sentiment_summary = {
    method: {
        lang: (
            df.groupby("dominant_topic")
            .agg(
                documents=("document", "count"),
                avg_topic_strength=("topic_strength", "mean"),
                avg_sentiment=("sentiment_score", "mean"),
                positive=("sentiment_label", lambda x: (x == "positive").sum()),
                neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
                negative=("sentiment_label", lambda x: (x == "negative").sum())
            )
            .reset_index()
        )
        for lang, df in lang_results.items()
    }
    for method, lang_results in doc_topics_sentiment.items()
}

# 5. Ausgabe: Topic + Sentiment je Modell und Sprache
for lang in ["de", "en"]:

    if lang == "de":
        pos_threshold, neg_threshold = 0.5, -0.5
    elif lang == "en":
        pos_threshold, neg_threshold = 0.1, -0.1

    for method_key, config in methods.items():
        display(
            style_topic_sentiment(
                topic_sentiment_summary[method_key][lang],
                f"{config['title']} ({lang}) - Topic & Sentiment Matrix",
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold
            )
        )
# -





# +
##########
# Coherence je Modell/Sprache berechnen
##########

coherence_results = []

for method_key, config in methods.items():
    for lang, result in config["topics_by_lang"].items():

        texts = tokenize_docs(result["documents"])

        topics = get_topics_from_model(
            model=result["model"],
            feature_names=result["features"],
            method_type=config["method_type"],
            top_n=10
        )

        coherence_score = compute_pmi_coherence(
            topics=topics,
            texts=texts
        )

        coherence_results.append({
            "method": method_key,
            "title": config["title"],
            "lang": lang,
            "coherence": coherence_score
        })

        print(
            f"{config['title']} ({lang}) "
            f"Coherence: {coherence_score:.4f}"
        )


coherence_df = pd.DataFrame(coherence_results)


# +
def plot_top_words_from_methods(methods, method_key, lang, topic_id, n_words=10):
    data = methods[method_key]["topics_by_lang"][lang]

    model = data["model"]
    features = data["features"]

    weights = model.components_[topic_id]
    top_idx = weights.argsort()[-n_words:][::-1]

    words = [features[i] for i in top_idx]
    values = [weights[i] for i in top_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(words[::-1], values[::-1])
    plt.title(f'{methods[method_key]["title"]} ({lang}) - Topic {topic_id + 1}')
    plt.xlabel("Gewichtung")
    plt.tight_layout()
    plt.show()

plot_top_words_from_methods(methods, "bow_lda", "de", topic_id=0)
plot_top_words_from_methods(methods, "tfidf_lsa", "de", topic_id=0)
# -



# +
def plot_topic_distribution_from_methods(methods, method_key, lang):
    data = methods[method_key]["topics_by_lang"][lang]
    doc_topic_matrix = data["doc_topic_matrix"]

    if methods[method_key]["method_type"] == "lsa":
        doc_topic_matrix = np.abs(doc_topic_matrix)

    dominant_topics = doc_topic_matrix.argmax(axis=1)

    counts = (
        pd.Series(dominant_topics)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title(f'{methods[method_key]["title"]} ({lang}) - Topic Verteilung')
    plt.xlabel("Topic")
    plt.ylabel("Anzahl Dokumente")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_topic_distribution_from_methods(methods, "tfidf_lsa", "de")
plot_topic_distribution_from_methods(methods, "tfidf_lsa", "en")


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
def compare_methods_topic_distribution(methods, lang):
    plt.figure(figsize=(10, 5))

    for method_key, config in methods.items():
        data = config["topics_by_lang"][lang]
        doc_topic_matrix = data["doc_topic_matrix"]

        if config["method_type"] == "lsa":
            doc_topic_matrix = np.abs(doc_topic_matrix)

        dominant_topics = doc_topic_matrix.argmax(axis=1)
        counts = pd.Series(dominant_topics).value_counts().sort_index()

        plt.plot(counts.index, counts.values, marker="o", label=config["title"])

    plt.title(f"Methodenvergleich - Topic Verteilung ({lang})")
    plt.xlabel("Topic")
    plt.ylabel("Anzahl Dokumente")
    plt.legend()
    plt.tight_layout()
    plt.show()

compare_methods_topic_distribution(methods, "de")
compare_methods_topic_distribution(methods, "en")
# -


