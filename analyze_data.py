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
from tqdm import tqdm #Progress Indication
from pprint import pprint,pformat #Saubere JSON darstellung

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
def main():

    # To be filled   
    print("To be filled")

if __name__ == "__main__":
    main()
# +
### Style Funktionen #######

def base_table_style(df, caption=None, extra_styles=None):
    styles = [
        {
            "selector": "table",
            "props": [
                ("table-layout", "fixed"),
                ("width", "100%")
            ]
        },
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("font-weight", "bold")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("white-space", "normal"),
                ("overflow-wrap", "break-word"),
                ("word-wrap", "break-word"),
                ("padding", "4px"),
                ("border-right", "1px solid lightgray")
            ]
        },
        {
            "selector": "caption",
            "props": [
                ("caption-side", "top"),
                ("text-align", "center"),
                ("font-weight", "bold"),
                ("font-size", "16px")
            ]
        }
    ]

    if extra_styles:
        styles.extend(extra_styles)

    styled = (
        df.style
        .hide(axis="index")
        .set_table_styles(styles)
        .set_properties(**{"text-align": "left"})
    )

    if caption:
        styled = styled.set_caption(caption)

    return styled

def style_topic_comparison(df, caption=None):
    method_start_positions = [
        pos for pos, value in enumerate(df["Method"])
        if value != ""
    ]

    def add_separator(row):
        pos = df.index.get_loc(row.name)

        if pos in method_start_positions and pos != 0:
            return ["border-top: 3px solid black"] * len(row)

        return [""] * len(row)

    return (
        base_table_style(
            df,
            caption=caption
        )
        .apply(add_separator, axis=1)
    )
    
def style_compare_top_tokens(df, caption=None):
    count_cols = [
        ("de", "Topic", "Count"),
        ("de", "Sentiment", "Count"),
        ("en", "Topic", "Count"),
        ("en", "Sentiment", "Count"),
    ]

    extra_styles = [
        {"selector": "th.col1, td.col1", "props": [("border-right", "1px solid black")]},
        {"selector": "th.col3, td.col3", "props": [("border-right", "4px solid black")]},
        {"selector": "th.col5, td.col5", "props": [("border-right", "1px solid black")]},
    ]

    return (
        base_table_style(df, caption, extra_styles)
        .set_properties(subset=count_cols, **{"text-align": "right"})
        .background_gradient(subset=count_cols, cmap="Blues")
    )

def style_topic_sentiment(
    df,
    caption,
    pos_threshold=0.3,
    neg_threshold=-0.3,
    pos_color="#c6efce",
    neg_color="#ffc7ce"
):
    """
    Styled Topic/Sentiment Tabelle mit:
    - Caption
    - konfigurierbaren Schwellenwerten
    - konfigurierbaren Farben
    """

    #caption = f"{method} ({lang}) – Topic-Verteilung & Sentiment"

    # --- Sentiment Highlight ---
    def highlight_sentiment(val):
        if val >= pos_threshold:
            return f"background-color: {pos_color}"
        elif val <= neg_threshold:
            return f"background-color: {neg_color}"
        return ""

    # --- Dominantes Sentiment hervorheben ---
    def highlight_dominance(row):
        max_val = max(row["positive"], row["neutral"], row["negative"])
        return [
            "font-weight: bold" if v == max_val else ""
            for v in [row["positive"], row["neutral"], row["negative"]]
        ]

    styled = base_table_style(df, caption=caption)

    styled = styled.map(highlight_sentiment, subset=["avg_sentiment"])

    styled = styled.apply(
        highlight_dominance,
        axis=1,
        subset=["positive", "neutral", "negative"]
    )
        # Kein Umbruch für Keywords
    if "topic_keywords" in df.columns:
        styled = styled.set_properties(
            subset=["topic_keywords"],
            **{
                "white-space": "nowrap",
                "word-wrap": "normal",
                "overflow-wrap": "normal",
                "min-width": "800px"
            }
        )

    return styled


# +
#### Analyse Funktionen #######
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


# +
def batch_lemmatizer(texts, nlp, stopw, desc, n_process):
    sentences = []

    reviews = nlp.pipe(texts, batch_size=200, n_process=n_process)

    for review in tqdm(reviews, total=len(texts), desc=desc):
        lemmas = [
            token.lemma_.lower()
            for token in review
            if token.is_alpha
            and len(token) > 2
            and not token.is_stop
            and token.lemma_.lower() not in stopw
        ]

        if lemmas:
            sentences.append(lemmas)

    return sentences
            
def clean_and_tokenize(df, stopw_de, stopw_en, n_process):
    stopw_de = set(stopw_de)
    stopw_en = set(stopw_en)

    nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    df_de = df[df["language"] == "de"]["review"].fillna("").astype(str)
    df_en = df[df["language"] == "en"]["review"].fillna("").astype(str)

    result = {}

    for lang, texts, nlp, stopw, desc in [
        ("de", df_de, nlp_de, stopw_de, "Deutsch verarbeiten"),
        ("en", df_en, nlp_en, stopw_en, "Englisch verarbeiten"),
    ]:
        sentences = batch_lemmatizer(
            texts=texts,
            nlp=nlp,
            stopw=stopw,
            desc=desc,
            n_process=n_process
        )

        documents = [" ".join(sentence) for sentence in sentences]
        vocabulary = sorted(set(word for sent in sentences for word in sent))
        index = {word: i for i, word in enumerate(vocabulary)}

        result[lang] = {
            "sentences": sentences,
            "documents": documents,
            "vocabulary": vocabulary,
            "index": index
        }

    return result

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
# Variable Definition
##########
local_dir = "data"
path_reviews = f"{local_dir}/combined_reviews.csv"

# Manuelle Stopwörter (werden iterativ ergänzt) 
stopw_topic_manual_de, stopw_sentiment_manual_de = [],[]
stopw_topic_manual_en, stopw_sentiment_manual_en = [],[]

# Stoppwörter aus NTLK
stopw_de = stopwords.words("german")
stopw_en = stopwords.words("english") 

# Zusammengesetzte Stopwörter aus manuell und NTLK
stopw_topic_de = stopw_de + stopw_topic_manual_de
stopw_topic_en = stopw_en + stopw_topic_manual_en
stopw_sentiment_de = stopw_de + stopw_sentiment_manual_de
stopw_sentiment_en = stopw_en + stopw_sentiment_manual_en

############
# Import data to pandas dataframe
############
df_reviews = pd.read_csv(path_reviews)

df_reviews["language"] = df_reviews["origin"].map({
    "FragdenStaat": "de",
    "YELP": "en",
})

############
# Prepare Data
############

# --- Stopwort-Definitionen ---
stopword_updates = {
    "iter_1": {
        "de_common": [
            "august","information","art","aktuell","zahl","genannt","frage","antwort","fall",
            "stelle","rahmen","bitten","insbesondere","erachten","tatsächlich",
            "bayerisch","bayern","baydsg","bayuig","vig", # Verwaltungsbegriffe
            "lebensmittelbetriebe","routinekontrolle","rüb","avv","lfgb" # Standardanfragen
        ],
        "en_common": [
            "want","look","know","think","people","day","find","tell","ask","take","work"
        ],
        "en_topic_only": [
            "nice","well","delicious","friendly","definitely"
        ]
    },

    "iter_2": {
        "de_common": [
            "bitte","einschließlich","datum","folgend","liegen","antrag","behörde","anfrage",
            "projekt","dokument","erfolgen","zuständig","zuständigkeitsbereich","falls",
            "auskunft","entsprechend","befinden","angabe","betreffen",
            "elektronisch","öffentlich","soweit","überprüfen","registriert", # Juristisch
            "aktenauskunft","gesetz","umweltinformationsgesetz" # Juristisch
        ],
        "en_common": [
            "place","food","get","try","time","come","go",
            "need","way","say","staff","experience"
        ],
        "en_topic_only": [
            "good","great","like","love","little"
        ]
    },

    "iter_3": { 
        "de_common": [ # Hauptsächlich Verwaltungsfloskeln 
            "letzter","form","begründung","interesse",
            "sämtlicher","anzahl","gemeinde","handeln",
            "geplant","sinn","monat", "freundlich",
            "senden","mitteilen","grüße","häufig"
            "stellen","höhe","angeben"
        ],
        "en_common": [ 
            "lot","feel","thing","visit", "pretty",
            "long","area","minute","new"
        ],
        "en_topic_only": ["bad","amazing"]
    },
    
    "iter_4": { 
        "de_common": [ 
            "unverzüglich", "ausdrücklich", "häufig", 
            "vorab", "bewerten", "übersicht", "jährlich", "mühe",
            "danken", "verweisen", "zugänglich"
        ],
        "en_common": [ 
            "sure", "right", "review", "hour","leave"
        ],
        "en_topic_only": ["recommend", "enjoy"]
    },
    
    "iter_5": { # Nach BoW + LDA Topic Modeling
        "de_common": [ 
            "satz", "empfangsbestätigung", "widersprechen", "weiterzuleiten",
            "unterrichten", "weitergabe", "aufwand", "gebührenpflichtig",
            "herr", "geehrt", "dame", "gemäß", "beantragen","gmbh",
            "vorhanden", "vorliegen", "zugang", "angefragt",
            "gewähren", "fragdenstaat", "verfahren", "gesetzlich",
            "grund", "bezug", "mitteilung"
        ],
        "en_common": [],
        "en_topic_only": []
    },

    "iter_6": { # Nach 2tem BoW + LDA Topic Modeling Durchlauf
        "de_common": [ 
            "einfach", "somit", "spätestens", "stellen",
            "zusätzlich", "sofern", "aufgrund"
        ],
        "en_common": [],
        "en_topic_only": []
    }
}

for iteration in stopword_updates.values():

    # Deutsch (immer beide)
    stopw_topic_de += iteration["de_common"]
    stopw_sentiment_de += iteration["de_common"]

    # Englisch (gemeinsam)
    stopw_topic_en += iteration["en_common"]
    stopw_sentiment_en += iteration["en_common"]

    # Englisch (nur Topic)
    stopw_topic_en += iteration["en_topic_only"]

# ggf. Duplikate entfernen
stopw_topic_de = sorted(set(stopw_topic_de))
stopw_sentiment_de = sorted(set(stopw_sentiment_de))
stopw_topic_en = sorted(set(stopw_topic_en))
stopw_sentiment_en = sorted(set(stopw_sentiment_en))


# Daten für Topic Modeling
print("Cleaning Data - Topic Modeling")
data_topic = clean_and_tokenize(
    df_reviews,
    stopw_de=stopw_topic_de,
    stopw_en=stopw_topic_en,
    n_process=2 # ggf. Anpassen für schnellere Verarbeitung
)

# Daten für Sentimentanalyse
print("\nCleaning Data - Sentiment Analysis")
data_sentiment = clean_and_tokenize(
    df_reviews,
    stopw_de=stopw_sentiment_de,
    stopw_en=stopw_sentiment_en,
    n_process=2
)



# -


top_n = 15
compare_top_tokens_table = compare_top_tokens(data_topic, data_sentiment, top_n)
display(style_compare_top_tokens(compare_top_tokens_table, caption=f"Top {top_n} Tokens: Topic vs Sentiment (de/en)"))

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

        # LDA-Modell trainieren
        if config["method_type"] == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method="online",
                max_iter=max_iter,
                evaluate_every=-1
            )

            # Manuelles Online-Training über mehrere Iterationen und Batches
            for _ in tqdm(range(max_iter), desc=f'{config["title"]} Topic tranieren ({lang})'):
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
                negative=("sentiment_label", lambda x: (x == "negative").sum()),
                topic_keywords=("topic_keywords", "first")
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

display(style_topic_comparison(topic_table_de, "Topic Vergleich - DE"))
print()
display(style_topic_comparison(topic_table_en, "Topic Vergleich - EN"))
# -



