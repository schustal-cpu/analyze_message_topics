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
    # n_process ermöglicht optionale Parallelisierung
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

def create_vectors(vector_type, documents, languages=None, separate_by_language=False):
    """
    Erzeugt Vektorrepräsentationen von Textdokumenten als Bag-of-Words
    oder TF-IDF. Optional können getrennte Matrizen je Sprache erzeugt werden.

    Parameter:
        vector_type (str): "bow" oder "tf-idf"
        documents (list): Liste von Dokumenten (Strings)
        languages (list, optional): Sprachlabels je Dokument, z. B. "de", "en"
        separate_by_language (bool): wenn True, separate Vektorisierung je Sprache

    Rückgabe:
        Bei separate_by_language=False:
            matrix, vectorizer, feature_names

        Bei separate_by_language=True:
            dict mit Sprache als Key und Matrix/Vectorizer/Features als Value
    """

    params = {
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.8,
        "dtype": np.float32,
    }

    def build_vectorizer():
        if vector_type == "bow":
            return CountVectorizer(**params)
        elif vector_type == "tf-idf":
            return TfidfVectorizer(**params)
        else:
            raise ValueError("vector_type muss 'bow' oder 'tf-idf' sein")

    # Standardfall: gemeinsamer Korpus
    if not separate_by_language:
        vectorizer = build_vectorizer()
        matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        return matrix, vectorizer, feature_names

    # Sprachgetrennter Korpus
    if languages is None:
        raise ValueError("Für separate_by_language=True muss languages übergeben werden.")

    results = {}

    for lang in sorted(set(languages)):
        docs_lang = [
            doc for doc, doc_lang in zip(documents, languages)
            if doc_lang == lang
        ]

        vectorizer = build_vectorizer()
        matrix = vectorizer.fit_transform(docs_lang)
        feature_names = vectorizer.get_feature_names_out()

        results[lang] = {
            "matrix": matrix,
            "vectorizer": vectorizer,
            "feature_names": feature_names,
            "documents": docs_lang,
        }

        print(f"{lang}: {matrix.shape[0]} Dokumente, {matrix.shape[1]} Features")

    return results


# +
# Variable Definition
local_dir = "data"
path_reviews = f"{local_dir}/combined_reviews.csv"

# Manuelle Stopwärter durch 
# token_counts = Counter(word for sent in sentences for word in sent) 
# print(token_counts.most_common(50))

stopw_manual_de = [
    "bitte", "einschließlich", "folgend", "datum", "falls", "soweit",
    "antrag", "anfrage", "behörde", "auskunft", "dokument",
    "öffentlich", "elektronisch", "zuständig", "zuständigkeitsbereich",
    "projekt", "angabe", "betreffen", "entsprechend",
    "liegen", "erfolgen", "befinden"
]

stopw_manual_en = [
    "good", "great", "place", "food", "like", "come",
    "go", "get", "try", "love", "time"
]

stopw_en = stopwords.words("english") + stopw_manual_en
stopw_de = stopwords.words("german") + stopw_manual_de

# Import data to pandas dataframe
df_reviews = pd.read_csv(path_reviews)

df_reviews["language"] = df_reviews["origin"].map({
    "FragdenStaat": "de",
    "YELP": "en",
})

sentences, documents, vocabulary, index = clean_and_tokenize(
    df_reviews,
    stopw_de=stopw_de,
    stopw_en=stopw_en,
    n_process=1  # bei vielen Daten ggf. 2 oder 4 testen
)

print(f"Anzahl Sätze: {len(sentences)}")
print(f"Vokabulargröße: {len(vocabulary)}")




# +
import random

print(f"Anzahl Dokumente: {len(documents)}")

# TF-IDF gemeinsam
tfidf_all_matrix, _, tfidf_all_features = create_vectors("tf-idf", documents)

# TF-IDF getrennt
tfidf_by_lang = create_vectors(
    "tf-idf",
    documents,
    languages=df_reviews["language"].tolist(),
    separate_by_language=True
)


def top_terms(matrix, features, doc_index, top_n=8):
    row = matrix[doc_index]
    top_idx = row.indices[row.data.argsort()[::-1][:top_n]]
    return [features[i] for i in top_idx]


# 10 zufällige Dokumente
sample_indices = random.sample(range(len(documents)), 10)

for i in sample_indices:
    lang = df_reviews.iloc[i]["language"]

    # Index im separierten Korpus bestimmen
    idx_lang = sum(df_reviews.iloc[:i]["language"] == lang)

    print(f"\n{'='*60}")
    print(f"Dokument {i} | Sprache: {lang}")
    print(f"{'-'*60}")
    
    print("ORIGINAL (gekürzt):")
    print(df_reviews.iloc[i]["review"][:200].replace("\n", " "))

    print("\nTF-IDF (gesamt):")
    print(top_terms(tfidf_all_matrix, tfidf_all_features, i))

    print(f"\nTF-IDF (separiert - {lang}):")
    print(top_terms(
        tfidf_by_lang[lang]["matrix"],
        tfidf_by_lang[lang]["feature_names"],
        idx_lang
    ))

# +
comparison, bow_df, tfidf_df = compare_bow_tfidf(
    documents,
    doc_index=2,
    top_n=20
)

print(comparison)
# -

tfidf_vectorizer.get_feature_names_out()
#tfidf_vectorizer.idf_

# +
# 1. Stichproben: Originaltext + Ergebnis vergleichen
#for i in range(10):
#    print("ORIGINAL:")
#    print(df_reviews.iloc[i]["review"][:500])
#    print("\nLEMMAS:")
#    print(sentences[i][:50])
#    print("-" * 80)

# 2. Häufigste Tokens prüfen

token_counts = Counter(word for sent in documents for word in sent)

print(token_counts.most_common(50))
# -

for i in range(10):
    print(len(df_reviews.iloc[i]["review"].split()))
    print(len(sentences[i]))


