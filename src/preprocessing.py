# src/preprocessing.py
import os
import json
import numpy as np

import spacy # Lemmatizer
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm #Progress Indication

def load_stopw_from_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
        
def write_stopw_to_json(dict_stopw, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict_stopw, file, ensure_ascii=False, indent=4)

def get_current_iteration_key(dict_stopw):
    iter_numbers = [
        int(k.replace("iter_", ""))
        for k in dict_stopw.keys()
        if k.startswith("iter_") and k.replace("iter_", "").isdigit()
    ]
    return f"iter_{max(iter_numbers)}"

def add_iter_to_json(dict_stopw, path):

    iter_numbers = [
        int(key.replace("iter_", ""))
        for key in dict_stopw.keys()
        if key.startswith("iter_") and key.replace("iter_", "").isdigit()
    ]

    current_iter = f"iter_{max(iter_numbers)}" if iter_numbers else "iter_0"

    next_iter = max(iter_numbers) + 1 if iter_numbers else 1
    next_key = f"iter_{next_iter}"

    if next_key not in dict_stopw:
        dict_stopw[next_key] = {
            "de_custom_common": [],
            "de_custom_topic_only": [],
            "en_custom_common": [],
            "en_custom_topic_only": []
        }

    write_stopw_to_json(dict_stopw, path)

    return dict_stopw, current_iter, next_key

def build_stopword_lists_from_iterations(dict_stopw):
    stopw_topic_de = []
    stopw_sentiment_de = []
    stopw_topic_en = []
    stopw_sentiment_en = []

    for iter_key, iteration in dict_stopw.items():

        if iter_key == "iter_0":
            stopw_topic_de += iteration.get("de_nltk_common", [])
            stopw_sentiment_de += iteration.get("de_nltk_common", [])
            
            stopw_topic_en += iteration.get("en_nltk_common", [])
            stopw_sentiment_en += iteration.get("en_nltk_common", [])

            stopw_topic_de += iteration.get("de_vader_topic_only", [])
            stopw_topic_en += iteration.get("en_vader_topic_only", [])

        else:
            # common = Topic UND Sentiment
            stopw_topic_de += iteration.get("de_custom_common", [])
            stopw_sentiment_de += iteration.get("de_custom_common", [])

            stopw_topic_en += iteration.get("en_custom_common", [])
            stopw_sentiment_en += iteration.get("en_custom_common", [])

            # topic_only = nur Topic Modeling
            stopw_topic_de += iteration.get("de_custom_topic_only", [])
            stopw_topic_en += iteration.get("en_custom_topic_only", [])

    return {
        "topic_de": sorted(set(stopw_topic_de)),
        "sentiment_de": sorted(set(stopw_sentiment_de)),
        "topic_en": sorted(set(stopw_topic_en)),
        "sentiment_en": sorted(set(stopw_sentiment_en)),
    }

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

def load_gervader_words(filepath, threshold):
    words = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            word = parts[0].lower()
            score = float(parts[1])

            if abs(score) >= threshold:
                words.append(word)

    return words

def load_vader_words(analyzer, threshold=1.5):
    
    words = []

    for word, score in analyzer.lexicon.items():
        if abs(score) >= threshold:
            words.append(word.lower())

    return words

def remove_duplicate_texts(df, text_col="document"):
    """
    Entfernt doppelte Texte aus einem DataFrame.

    Parameter:
        df (pd.DataFrame): Input DataFrame
        text_col (str): Spalte mit Text

    Rückgabe:
        df_clean (pd.DataFrame): DataFrame ohne Duplikate
    """

    if text_col not in df.columns:
        raise ValueError(f"Spalte '{text_col}' nicht im DataFrame gefunden.")

    # Größe vorher
    before = len(df)

    # Normalisierung (wichtig!)
    df["_norm_text"] = (
        df[text_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Duplikate entfernen
    df_clean = df.drop_duplicates(subset="_norm_text").copy()

    # Hilfsspalte entfernen
    df_clean = df_clean.drop(columns=["_norm_text"])

    # Größe nachher
    after = len(df_clean)

    print("\n=== Duplikat-Entfernung ===")
    print(f"Vorher: {before} Zeilen")
    print(f"Nachher: {after} Zeilen")
    print(f"Entfernt: {before - after} Duplikate")

    return df_clean

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