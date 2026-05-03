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
import re
from pprint import pprint,pformat #Saubere JSON darstellung
import os
import json

import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from collections import Counter

import spacy # Lemmatizer

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GerVADER.vaderSentimentGER import SentimentIntensityAnalyzer as GerSentimentIntensityAnalyzer

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src.styles import *
from src.visualization import *
from src.preprocessing import *

import ipywidgets as widgets


# -

# **Beschriebenes Vorgehen:**
#
# Bereinigung durch:
# - Pandas
# - NLTK
# - SpaCy (besser für Deutsches Lemmatizing)
#
# -> Separat für Topic Modeling/ Semantik Analyse
#
#
# Vektorisierung anhand BoW und TF-IDF + n-Gramme durch:
# - sklearn
#
# -> Darstellung Unterschied zwischen BoW u TF-IDF
# -> TF-IDF sprachsepariert, da sonst falsche stoppwörter
#
#
# Themenidentifikation (LSA/LDA) durch:
# - sklearn
# - vaderSentiment
# - GerVADER
#
# -> Ausarbeitung bester Ansatz:
#
# BoW + LDA
# TF-IDF + LDA
# -> TF-IDF + LSA
#
#

# +
def vectorize(Vectorizer, data_by_lang):
    params = {
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.8,
        "dtype": np.float32,
    }

    results = {}

    for lang, data in data_by_lang.items():
        documents = data["documents"]

        vectorizer = Vectorizer(**params)
        matrix = vectorizer.fit_transform(documents)

        results[lang] = {
            "matrix": matrix,
            "vectorizer": vectorizer,
            "feature_names": vectorizer.get_feature_names_out(),
            "documents": documents,
            "sentences": data["sentences"],
            "vocabulary": data["vocabulary"],
            "index": data["index"]
        }

        print(f"{lang}: {matrix.shape[0]} Dokumente, {matrix.shape[1]} Features")

    return results

def assign_topics_with_keywords(model_result, method, n_top_words=10):
    """
    Einheitliche Topic-Zuordnung für LDA und LSA inkl. Top-Keywords.

    Erwartet in model_result:
        - "doc_topic_matrix"
        - "documents"
        - "model"
        - "features"

    Parameter:
        method (str): 'lda' oder 'lsa'
        n_top_words (int): Anzahl Top-Wörter pro Topic

    Rückgabe:
        DataFrame mit:
        - document
        - dominant_topic
        - topic_strength
        - topic_keywords
    """

    doc_topic_matrix = model_result["doc_topic_matrix"]

    # --- Topic-Zuordnung ---
    if method == "lda":
        dominant_topics = doc_topic_matrix.argmax(axis=1)
        topic_strengths = doc_topic_matrix.max(axis=1)

    elif method == "lsa":
        dominant_topics = np.abs(doc_topic_matrix).argmax(axis=1)
        topic_strengths = np.abs(doc_topic_matrix).max(axis=1)

    else:
        raise ValueError("method muss 'lda' oder 'lsa' sein")

    # --- Keywords pro Topic ---
    model = model_result["model"]
    features = model_result["features"]

    topic_keywords = {
        i + 1: ", ".join(
            features[j]
            for j in topic.argsort()[-n_top_words:][::-1]
        )
        for i, topic in enumerate(model.components_)
    }

    # --- DataFrame ---
    df = pd.DataFrame({
        "document": model_result["documents"],
        "dominant_topic": dominant_topics + 1,
        "topic_strength": topic_strengths
    })

    # Keywords mappen
    df["topic_keywords"] = df["dominant_topic"].map(topic_keywords)

    return df

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



# +
##########
# Data Import
##########

# Variable Definition
local_dir = "data"
path_reviews = f"{local_dir}/combined_reviews.csv"
stopword_path = "data/stopwords_custom.json"

top_n = 15
n_process = 1

# Import data to pandas dataframe
df_reviews = pd.read_csv(path_reviews)

df_reviews["language"] = df_reviews["origin"].map({
    "FragdenStaat": "de",
    "YELP": "en",
})

##########
# Stopword Definition
##########

# Basis-Dictionary
dict_stopw_default = {
    "iter_0": {
        "de_nltk_common": stopwords.words("german"),
        "en_nltk_common": stopwords.words("english")
    },
    "iter_1": {
        "de_custom_common": [],
        "de_custom_topic_only": [],
        "en_custom_common": [],
        "en_custom_topic_only": []
    }
}


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

    stopw_topic_de = stopwords_all["topic_de"]
    stopw_sentiment_de = stopwords_all["sentiment_de"]

    stopw_topic_en = stopwords_all["topic_en"]
    stopw_sentiment_en = stopwords_all["sentiment_en"]

    # Cleaning Topic
    print("\nCleaning Data - Topic Modeling")
    data_topic = clean_and_tokenize(
        df_reviews,
        stopw_de=stopw_topic_de,
        stopw_en=stopw_topic_en,
        n_process=n_process
    )

    # Cleaning Sentiment
    print("\nCleaning Data - Sentiment Analysis")
    data_sentiment = clean_and_tokenize(
        df_reviews,
        stopw_de=stopw_sentiment_de,
        stopw_en=stopw_sentiment_en,
        n_process=n_process
    )

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
        dict_stopw = add_iter_to_json(stopword_path)
    
    elif choice == "2":
        print("\nFinale Stopwörter übernommen. Weiter mit Topic Modeling.")
        break
    
    else:
        print("\nUngültige Eingabe. Bitte 1 oder 2 wählen.")

# +
############
# Create Vectors with BoW & TF-IDF
# Separate Pipelines: Topic Modeling vs. Sentiment
############

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
# Topic Modeling
############

# Zentrale Parameter für alle Topic-Modelle
n_topics = 10          # Anzahl der Topics
n_top_words = 15       # Anzahl der Top-Wörter pro Topic
max_iter = 20          # Anzahl der Trainingsdurchläufe für LDA
batch_size = 500       # Batch-Größe für das Online-Learning bei LDA

# Zentrale Konfiguration aller Topic-Modelle
methods = {
    "bow_lda": {
        "title": "LDA BoW",
        "vectors_by_lang": vectors_topic["bow_by_lang"],
        "method_type": "lda",
    },
    "tfidf_lda": {
        "title": "LDA TF-IDF",
        "vectors_by_lang": vectors_topic["tfidf_by_lang"],
        "method_type": "lda",
    },
    "tfidf_lsa": {
        "title": "LSA TF-IDF",
        "vectors_by_lang": vectors_topic["tfidf_by_lang"],
        "method_type": "lsa",
    },
}

# Jedes konfigurierte Modell trainieren
for method_key, config in methods.items():

    # Hier werden die trainierten Ergebnisse je Sprache gespeichert
    config["topics_by_lang"] = {}

    # Sprachgetrenntes Training, für "de" und "en"
    for lang, data in config["vectors_by_lang"].items():
        X = data["matrix"]  # Dokument-Term-Matrix der jeweiligen Sprache

        # LDA-Modell trainieren - Für BoW und TF-IDF mit den selben Werten
        if config["method_type"] == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method="online",
                max_iter=max_iter,
                evaluate_every=-1
            )

            # Manuelles Online-Training über mehrere Iterationen und Batches
            for _ in tqdm(range(max_iter), desc=f'{config["title"]} Topic trainieren ({lang})'):
                for i in range(0, X.shape[0], batch_size):
                    model.partial_fit(X[i:i + batch_size])

            # Dokumente in Topic-Wahrscheinlichkeiten transformieren
            topic_matrix = model.transform(X)

        # LSA-Modell trainieren
        elif config["method_type"] == "lsa":
            model = TruncatedSVD(
                n_components=n_topics,
                random_state=42
            )

            # Dokumente direkt in latente Topic-Komponenten transformieren
            topic_matrix = model.fit_transform(X)
            print(f'{config["title"]} Topic trainiert ({lang})')


        # Modell- und Ergebnisdaten zentral im methods-Dict speichern
        config["topics_by_lang"][lang] = {
            "model": model,
            "doc_topic_matrix": topic_matrix,
            "matrix": X,
            "features": data["feature_names"],
            "documents": data["documents"]
        }


# +
def get_structure(d, max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return "..."

    if isinstance(d, dict):
        return {
            k: get_structure(v, max_depth, current_depth + 1)
            for k, v in d.items()
        }

    return type(d).__name__


pprint(get_structure(next(iter(methods.values())), max_depth=3))

# +
############
# Topic + Sentiment Analyse
############


# 1. Topic-Zuordnung je Modell
doc_topics = {
    method: {
        lang: assign_topics_with_keywords(
            result,
            method=config["method_type"]
        )
        for lang, result in config["topics_by_lang"].items()
    }
    for method, config in methods.items()
}

# 2. Sentiment je Sprache berechnen
# Sentiment basiert auf den Sentiment-Dokumenten, nicht auf BoW/TF-IDF
sentiments = {
    lang: build_sentiment_df(result, lang=lang)
    for lang, result in vectors_sentiment["tfidf_by_lang"].items()
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
                avg_sentiment=("sentiment_score", "mean"),
                positive=("sentiment_label", lambda x: (x == "positive").sum()),
                neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
                negative=("sentiment_label", lambda x: (x == "negative").sum())
                #topic_keywords=("topic_keywords", "first")
            )
            .reset_index()
        )
        for lang, df in lang_results.items()
    }
    for method, lang_results in doc_topics_sentiment.items()
}


# +
# 5. Ausgabe: BoW/LDA Topic + Sentiment

for lang in ["de", "en"]:
    if lang == "de":
        pos_threshold, neg_threshold = 0.5, -0.5
    elif lang == "en":
        pos_threshold, neg_threshold = 0.1, -0.1

    for method_key, config in methods.items():
        display(style_topic_sentiment(
            topic_sentiment_summary[method_key][lang],
            f"{config["title"]} ({lang}) - Topic & Sentiment Matrix",
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold
        ))

# +
# 6. Ausgabe: Topics per Method/ Language
topic_table_de = build_topic_comparison_tables(doc_topics,methods,"de")
topic_table_en = build_topic_comparison_tables(doc_topics,methods,"en")

#display(style_topic_comparison(topic_table_de, "Topic Vergleich - DE"))
#print()
#display(style_topic_comparison(topic_table_en, "Topic Vergleich - EN"))


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
plot_top_words_from_methods(methods, "tfidf_lsa", "en", topic_id=2)


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

# +


def plot_topic_heatmap(methods, method_key, lang):
    data = methods[method_key]["topics_by_lang"][lang]
    matrix = data["doc_topic_matrix"]

    if methods[method_key]["method_type"] == "lsa":
        matrix = np.abs(matrix)

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix[:50], cmap="viridis")  # erste 50 Dokumente
    plt.title(f'{methods[method_key]["title"]} ({lang}) - Topic Heatmap')
    plt.xlabel("Topics")
    plt.ylabel("Dokumente")
    plt.tight_layout()
    plt.show()

plot_topic_heatmap(methods, "bow_lda", "de")
plot_topic_heatmap(methods, "tfidf_lsa", "en")

# +
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
import numpy as np

def evaluate_methods(methods, top_n_words=10):
    results = []

    for method_key, config in methods.items():
        method_type = config["method_type"]
        title = config["title"]

        for lang in ["de", "en"]:
            topic_data = config["topics_by_lang"][lang]
            vector_data = config["vectors_by_lang"][lang]

            model = topic_data["model"]
            X = topic_data["matrix"]
            features = topic_data["features"]
            texts = vector_data["sentences"]

            # --- Topics extrahieren ---
            topics = []
            for topic in model.components_:
                top_idx = topic.argsort()[-top_n_words:][::-1]
                topics.append([features[i] for i in top_idx])

            # --- Coherence ---
            dictionary = Dictionary(texts)
            coherence_model = CoherenceModel(
                topics=topics,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v"
            )
            coherence = coherence_model.get_coherence()

            # --- LDA spezifische Metriken ---
            if method_type == "lda":
                perplexity = model.perplexity(X)
                log_likelihood = model.score(X)
            else:
                perplexity = np.nan
                log_likelihood = np.nan

            results.append({
                "method": title,
                "lang": lang,
                "coherence": coherence,
                "perplexity": perplexity,
                "log_likelihood": log_likelihood
            })

    return pd.DataFrame(results)

eval_df = evaluate_methods(methods)
display(eval_df.sort_values(by="coherence", ascending=False))
# -


