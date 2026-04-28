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
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

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
# Vektorisierung anhand BoW und TF-IDF + n-Gramme durch:
# - sklearn
#
# Themenidentifikation (LSA/LDA) durch:
# - sklearn
# - vaderSentiment
# - GerVADER
#

# +
def main():

    # To be filled   
    print("To be filled")

if __name__ == "__main__":
    main()


# -
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

    print("\n=== 3. Häufigste n-Gramme (TF-IDF Basis) ===")
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8)
    X = vec.fit_transform(documents)

    freqs = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()

    df_ngrams = pd.DataFrame({
        "term": terms,
        "freq": freqs
    }).sort_values(by="freq", ascending=False)

    print(df_ngrams.head(top_n))

    print("\n=== 4. Dokumentfrequenz (für min_df/max_df tuning) ===")
    df_docfreq = (X > 0).sum(axis=0).A1
    df_df = pd.DataFrame({
        "term": terms,
        "doc_freq": df_docfreq
    }).sort_values(by="doc_freq", ascending=False)

    print(df_df.head(top_n))


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

def vectorize(Vectorizer, documents, languages=None, separate_by_language=False):
    params = {
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.8,
        "dtype": np.float32,
    }

    if not separate_by_language:
        vectorizer = Vectorizer(**params)
        matrix = vectorizer.fit_transform(documents)
        return matrix, vectorizer, vectorizer.get_feature_names_out()

    if languages is None:
        raise ValueError("Für separate_by_language=True muss languages übergeben werden.")

    results = {}

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

def create_vector_sets(documents, languages):
    bow_matrix, bow_vectorizer, bow_features = vectorize(
        CountVectorizer,
        documents
    )

    tfidf_by_lang = vectorize(
        TfidfVectorizer,
        documents,
        languages=languages,
        separate_by_language=True
    )

    return {
        "bow": {
            "matrix": bow_matrix,
            "vectorizer": bow_vectorizer,
            "features": bow_features
        },
        "tfidf_by_lang": tfidf_by_lang
    }
    
def print_lda_topics(model, feature_names, n_top_words=10):
    """
    Gibt die wichtigsten Wörter je Topic aus.
    """
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]

        print(f"\nTopic {topic_idx + 1}:")
        print(", ".join(top_words))



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
sentences_topic, documents_topic, vocabulary_topic, index_topic = clean_and_tokenize(
    df_reviews,
    stopw_de=stopw_topic_de,
    stopw_en=stopw_topic_en,
    n_process=2  # bei vielen Daten ggf. 2 oder 4 testen
)

# Daten für Sentimentanalyse
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

# +
############
# Create Vectors with BoW & TF-IDF
# Separate Pipelines: Topic Modeling vs. Sentiment
############

languages = df_reviews["language"].tolist()

vectors_topic = create_vector_sets(documents_topic, languages)
vectors_sentiment = create_vector_sets(documents_sent, languages)

bow_topic_matrix = vectors_topic["bow"]["matrix"]

tfidf_topic_matrix_de = vectors_topic["tfidf_by_lang"]["de"]["matrix"]
tfidf_topic_features_de = vectors_topic["tfidf_by_lang"]["de"]["feature_names"]
tfidf_topic_matrix_en = vectors_topic["tfidf_by_lang"]["en"]["matrix"]
tfidf_topic_features_en = vectors_topic["tfidf_by_lang"]["en"]["feature_names"]

tfidf_sentiment_matrix_de = vectors_sentiment["tfidf_by_lang"]["de"]["matrix"]
tfidf_sentiment_features_de = vectors_sentiment["tfidf_by_lang"]["de"]["feature_names"]
tfidf_sentiment_matrix_en = vectors_sentiment["tfidf_by_lang"]["en"]["matrix"]
tfidf_sentiment_features_en = vectors_sentiment["tfidf_by_lang"]["en"]["feature_names"]

# +
############
# Topic Modeling
############

# Erste Themenanalyse: Topic-Pipeline mit BoW + LDA
n_topics = 10
n_top_words = 15
max_iter = 20

lda_bow_topic = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method="online",
    max_iter=1,
    evaluate_every=-1
)

for _ in tqdm(range(max_iter), desc="LDA BoW Topic trainieren"):
    lda_bow_topic.partial_fit(vectors_topic["bow"]["matrix"])
# -

print_lda_topics(
    lda_bow_topic,
    vectors_topic["bow"]["features"],
    n_top_words=n_top_words
)


