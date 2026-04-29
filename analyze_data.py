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
from pprint import pprint #Saubere JSON darstellung

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

#import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick


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
#### Analyse Funktionen #######
def analyze_tokens(sentences, documents, languages=None, top_n=30):
    """
    Gibt relevante Statistiken zur Stopwort-Optimierung aus.
    """

    print("\n=== 1. Häufigste Tokens (gesamt) ===")
    token_counts = Counter(word for sent in sentences for word in sent)
    for word, count in token_counts.most_common(top_n):
        print(f"{word:20s} {count}")

    print("\n=== 2. Häufigste Tokens je Sprache ===")
    if languages is not None:
        df_tmp = pd.DataFrame({
            "doc": documents,
            "lang": languages
        })

        for lang in sorted(df_tmp["lang"].unique()):
            tokens_lang = [
                word
                for doc, l in zip(sentences, languages) if l == lang
                for word in doc
            ]
            counts_lang = Counter(tokens_lang)

            print(f"\n--- Sprache: {lang} ---")
            for word, count in counts_lang.most_common(top_n):
                print(f"{word:20s} {count}")

def compare_top_tokens(sentences_topic, sentences_sent, top_n=30):
    # Top Tokens Topic
    topic_top = pd.DataFrame(
        Counter(w for s in sentences_topic for w in s).most_common(top_n),
        columns=["topic_token", "topic_count"]
    )

    # Top Tokens Sentiment
    sent_top = pd.DataFrame(
        Counter(w for s in sentences_sent for w in s).most_common(top_n),
        columns=["sent_token", "sent_count"]
    )

    # Vergleich nebeneinander
    comparison = pd.concat([topic_top, sent_top], axis=1)

    display(comparison)
    return comparison

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



# +
def batch_lemmatizer(texts, nlp, stopw, desc, sentences, n_process):
    """
    Führt eine batchweise Lemmatisierung und Filterung von Texten durch.

    Parameter:
        texts (Iterable): Sammlung von Texten (z. B. pandas Series)
        nlp (spaCy Language): geladenes Sprachmodell (z. B. Deutsch/Englisch)
        stopw (set): Stopwortmenge für effiziente Filterung
        desc (str): Beschreibung für Fortschrittsanzeige (tqdm)
    """

    # Verarbeitung der Texte in Batches für erhöhte Effizienz
    # n_process ermöglicht zusätzlich Parallelisierung
    reviews = nlp.pipe(texts, batch_size=200, n_process=n_process)

    # Iteration über die verarbeiteten Reviews mit Fortschrittsanzeige
    for review in tqdm(reviews, total=len(texts), desc=desc):

        # Lemmatisierung und Filterung:
        # - nur alphabetische Tokens (keine Zahlen/Sonderzeichen)
        # - Entfernen spaCy-interner Stopwörter
        # - zusätzliche Filterung über NTLK + manuelle Stopwortliste
        lemmas = [
            token.lemma_.lower()
            for token in review
            if token.is_alpha
            and len(token) > 2 # Nachträglich ergänzt, da überflüssige Artifakte entdeckt
            and not token.is_stop
            and token.lemma_.lower() not in stopw
        ]

        # Nur nicht-leere Ergebnisse in die globale Satzliste aufnehmen
        if lemmas:
            sentences.append(lemmas)
            
def clean_and_tokenize(df, stopw_de, stopw_en, n_process):
    """
    Führt eine sprachspezifische Textvorverarbeitung durch.
    
    Schritte:
    - Tokenisierung und Lemmatisierung (via spaCy)
    - Entfernung von Stopwörtern
    - Erstellung einer Satzliste (Token-Listen)
    - Aufbau eines Vokabulars und Index
    
    Parameter:
        df (DataFrame): Enthält mindestens 'review' und 'language'
        stopw_de (list): deutsche Stopwörter
        stopw_en (list): englische Stopwörter
        n_process (int): Anzahl paralleler Prozesse für spaCy

    Rückgabe:
        sentences (list): Liste von Token-Listen
        vocabulary (list): eindeutige Tokens
        index (dict): Mapping Token → Index
    """

    sentences = []

    # Umwandlung der Stopwortlisten in Sets für effiziente Lookups (O(1))
    stopw_de = set(stopw_de)
    stopw_en = set(stopw_en)

    # Laden der spaCy-Modelle (Parser und NER deaktiviert für Performance)
    nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Trennung der Daten nach Sprache
    df_de = df[df["language"] == "de"]["review"].fillna("").astype(str)
    df_en = df[df["language"] == "en"]["review"].fillna("").astype(str)

    # Sprachabhängige Lemmatisierung mittels spaCy
    batch_lemmatizer(df_de, nlp_de, stopw_de, "Deutsch verarbeiten", sentences, n_process)
    batch_lemmatizer(df_en, nlp_en, stopw_en, "Englisch verarbeiten", sentences, n_process)

    # Aufbau des Vokabulars:
    # Alle Tokens werden gesammelt und Duplikate entfernt (set)
    vocabulary = sorted(set(word for sent in sentences for word in sent))

    # Erstellung eines Index (Token → numerischer Index)
    index = {word: i for i, word in enumerate(vocabulary)}

    # document Format für weiterbearbeitung druch NTLK 
    documents = [" ".join(sentence) for sentence in sentences]

    return sentences, documents, vocabulary, index

def vectorize(Vectorizer, documents, languages):
    params = {
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.8,
        "dtype": np.float32,
    }

    results={}

    for lang in sorted(set(languages)):
        docs_lang = [doc for doc, l in zip(documents, languages) if l == lang]

        vectorizer = Vectorizer(**params)
        matrix = vectorizer.fit_transform(docs_lang)

        results[lang] = {
            "matrix": matrix,
            "vectorizer": vectorizer,
            "feature_names": vectorizer.get_feature_names_out(),
            "documents": docs_lang,
        }

        print(f"{lang}: {matrix.shape[0]} Dokumente, {matrix.shape[1]} Features")

    return results
    


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
sentences_topic, documents_topic, vocabulary_topic, index_topic = clean_and_tokenize(
    df_reviews,
    stopw_de=stopw_topic_de,
    stopw_en=stopw_topic_en,
    n_process=2  # bei vielen Daten ggf. 2 oder 4 testen
)

# Daten für Sentimentanalyse
print("\nCleaning Data - Sentiment Analysis")
sentences_sent, documents_sent, vocabulary_sent, index_sent = clean_and_tokenize(
    df_reviews,
    stopw_de=stopw_sentiment_de,
    stopw_en=stopw_sentiment_en,
    n_process=2
)


# -


analyze_tokens(
    sentences_topic,
    documents_topic,
    languages=df_reviews["language"].tolist(),
    top_n=30
)

comparison_tokens = compare_top_tokens(
    sentences_topic,
    sentences_sent,
    top_n=30
)

# +
############
# Create Vectors with BoW & TF-IDF
# Separate Pipelines: Topic Modeling vs. Sentiment
############

languages = df_reviews["language"].tolist()

print("Creating Vectors - Topic Modeling")
vectors_topic = {}
vectors_topic["bow_by_lang"] = vectorize(CountVectorizer, documents_topic, languages)
vectors_topic["tfidf_by_lang"] = vectorize(TfidfVectorizer, documents_topic, languages)

print("\nCreating Vectors - Sentiment Analysis")
vectors_sentiment = {}
vectors_sentiment["bow_by_lang"] = vectorize(CountVectorizer, documents_sent, languages)
vectors_sentiment["tfidf_by_lang"] = vectorize(TfidfVectorizer, documents_sent, languages)


# +
############
# Topic Modeling
############

n_topics = 10
n_top_words = 15
max_iter = 20

# --- BoW + LDA sprachgetrennt ---
lda_bow_topics_by_lang = {}

for lang, data in vectors_topic["bow_by_lang"].items():
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="online",
        max_iter=1,
        evaluate_every=-1
    )

    for _ in tqdm(range(max_iter), desc=f"LDA BoW Topic trainieren ({lang})"):
        lda_model.partial_fit(data["matrix"])

    lda_bow_topics_by_lang[lang] = {
        "model": lda_model,
        "matrix": data["matrix"],
        "features": data["feature_names"],
        "documents": data["documents"]
    }

# --- TF-IDF + LDA sprachgetrennt ---

lda_tfidf_topics_by_lang = {}

for lang, data in vectors_topic["tfidf_by_lang"].items():
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="online",
        max_iter=1,
        evaluate_every=-1
    )

    for _ in tqdm(range(max_iter), desc=f"LDA TF-IDF Topic trainieren ({lang})"):
        lsa_matrix = lda_model.partial_fit(data["matrix"])

    lda_tfidf_topics_by_lang[lang] = {
        "model": lda_model,
        "matrix": data["matrix"],
        "features": data["feature_names"]
    }

# --- TF-IDF + LSA sprachgetrennt ---
lsa_tfidf_topics_by_lang = {}

for lang, data in tqdm(vectors_topic["tfidf_by_lang"].items(), desc="LSA TF-IDF Topic trainieren (de/en)"):

    lsa_model = TruncatedSVD(
        n_components=n_topics,
        random_state=42
    )

    lsa_matrix = lsa_model.fit_transform(data["matrix"])

    lsa_tfidf_topics_by_lang[lang] = {
        "model": lsa_model,
        "matrix": lsa_matrix,
        "features": data["feature_names"]
    }


# +
df_bow_lda_en = topics_matrix(
    lda_bow_topics_by_lang["en"]["model"],
    vectors_topic["bow_by_lang"]["en"]["feature_names"],
    n_top_words=10
)

df_tfidf_lda_en = topics_matrix(
    lda_tfidf_topics_by_lang["en"]["model"],
    vectors_topic["tfidf_by_lang"]["en"]["feature_names"],
    n_top_words=10
)

df_tfidf_lsa_en = topics_matrix(
    lsa_tfidf_topics_by_lang["en"]["model"],
    vectors_topic["tfidf_by_lang"]["en"]["feature_names"],
    n_top_words=10
)


df_bow_lda_de = topics_matrix(
    lda_bow_topics_by_lang["de"]["model"],
    vectors_topic["bow_by_lang"]["de"]["feature_names"],
    n_top_words=10
)

df_tfidf_lda_de = topics_matrix(
    lda_tfidf_topics_by_lang["de"]["model"],
    vectors_topic["tfidf_by_lang"]["de"]["feature_names"],
    n_top_words=10
)

df_tfidf_lsa_de = topics_matrix(
    lsa_tfidf_topics_by_lang["de"]["model"],
    vectors_topic["tfidf_by_lang"]["de"]["feature_names"],
    n_top_words=10
)


from IPython.display import display, Markdown

# -------- Englisch --------
display(Markdown("## BoW + LDA (EN)"))
display(df_bow_lda_en)

display(Markdown("## TF-IDF + LDA (EN)"))
display(df_tfidf_lda_en)

display(Markdown("## TF-IDF + LSA (EN)"))
display(df_tfidf_lsa_en)


# -------- Deutsch --------
display(Markdown("## BoW + LDA (DE)"))
display(df_bow_lda_de)

display(Markdown("## TF-IDF + LDA (DE)"))
display(df_tfidf_lda_de)

display(Markdown("## TF-IDF + LSA (DE)"))
display(df_tfidf_lsa_de)

# +
models_topic_de = {
    "BoW/LDA (de)": {
        "model": lda_bow_topics_by_lang["de"]["model"],
        "features": lda_bow_topics_by_lang["de"]["features"]
    },

    "TF-IDF/LDA (de)": {
        "model": lda_tfidf_topics_by_lang["de"]["model"],
        "features": lda_tfidf_topics_by_lang["de"]["features"]
    },

    "TF-IDF/LSA (de)": {
        "model": lsa_tfidf_topics_by_lang["de"]["model"],
        "features": lsa_tfidf_topics_by_lang["de"]["features"]
    }
}

models_topic_en = {
    "BoW/LDA (en)": {
        "model": lda_bow_topics_by_lang["en"]["model"],
        "features": lda_bow_topics_by_lang["en"]["features"]
    },

    "TF-IDF/LDA (en)": {
        "model": lda_tfidf_topics_by_lang["en"]["model"],
        "features": lda_tfidf_topics_by_lang["en"]["features"]
    },

    "TF-IDF/LSA (en)": {
        "model": lsa_tfidf_topics_by_lang["en"]["model"],
        "features": lsa_tfidf_topics_by_lang["en"]["features"]
    }
}

df_topic_comparison_de = compare_topic_models_multiindex(
    models_topic_de,
    n_top_words
)

df_topic_comparison_en = compare_topic_models_multiindex(
    models_topic_en,
    n_top_words
)

display(df_topic_comparison_de)
display(df_topic_comparison_en)

# +
from IPython.display import display, Markdown

display(Markdown("## TF-IDF + LSA (EN)"))
display(df_tfidf_lsa_en)

display(Markdown("## TF-IDF + LSA (DE)"))
display(df_tfidf_lsa_de)


# +
def flatten_columns(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(map(str, col)).strip() for col in df.columns]
    return df

def highlight_duplicates(df):
    df = flatten_columns(df).astype(str).replace({"nan": "", "None": ""})

    counts = pd.Series(df.to_numpy().ravel())
    counts = counts[counts != ""].value_counts()

    def color_cell(value):
        freq = counts.get(str(value), 0)
        if freq >= 5:
            return "background-color: #f4b183"
        elif freq >= 3:
            return "background-color: #ffd966"
        elif freq >= 2:
            return "background-color: #fff2cc"
        return ""

    return df.style.map(color_cell)

def show_topic_tables(tables, language):
    display(Markdown(f"# Topic-Vergleich {language}"))

    for title, df in tables.items():
        display(Markdown(f"## {title}"))
        display(highlight_duplicates(df))

tables_de = {
    "BoW + LDA": df_bow_lda_de,
    "TF-IDF + LDA": df_tfidf_lda_de,
    "TF-IDF + LSA": df_tfidf_lsa_de
}

tables_en = {
    "BoW + LDA": df_bow_lda_en,
    "TF-IDF + LDA": df_tfidf_lda_en,
    "TF-IDF + LSA": df_tfidf_lsa_en
}

show_topic_tables(tables_de, "Deutsch")
show_topic_tables(tables_en, "Englisch")
# -


