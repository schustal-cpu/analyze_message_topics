# src/preprocessing.py

import os
import json

import spacy  # NLP pipeline for tokenization and lemmatization
from tqdm import tqdm  # Progress bar for long-running operations

from src.visualization import compare_top_tokens


def load_stopw_from_json(path):
    """
    Load stopword configuration from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Dictionary containing stopword iterations and categories.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_stopw_to_json(dict_stopw, path):
    """
    Write stopword dictionary to a JSON file.

    Ensures that the target directory exists before writing.

    Parameters
    ----------
    dict_stopw : dict
        Stopword dictionary to be saved.

    path : str
        Target file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict_stopw, file, ensure_ascii=False, indent=4)


def get_current_iteration_key(dict_stopw):
    """
    Retrieve the latest iteration key from the stopword dictionary.

    Iterations are expected to follow the naming convention 'iter_<number>'.

    Parameters
    ----------
    dict_stopw : dict
        Stopword dictionary containing iteration keys.

    Returns
    -------
    str
        Key of the latest iteration (e.g., 'iter_3').
    """
    iter_numbers = [
        int(k.replace("iter_", ""))
        for k in dict_stopw.keys()
        if k.startswith("iter_") and k.replace("iter_", "").isdigit()
    ]
    return f"iter_{max(iter_numbers)}"


def add_iter_to_json(dict_stopw, path):
    """
    Add a new iteration entry to the stopword JSON structure.

    The function determines the current iteration and creates the next one
    with empty custom stopword categories.

    Parameters
    ----------
    dict_stopw : dict
        Existing stopword dictionary.

    path : str
        File path where the updated dictionary should be saved.

    Returns
    -------
    tuple
        (updated_dict, current_iteration_key, next_iteration_key)
    """

    # Extract numeric iteration indices
    iter_numbers = [
        int(key.replace("iter_", ""))
        for key in dict_stopw.keys()
        if key.startswith("iter_") and key.replace("iter_", "").isdigit()
    ]

    # Determine current iteration
    current_iter = f"iter_{max(iter_numbers)}" if iter_numbers else "iter_0"

    # Determine next iteration number
    next_iter = max(iter_numbers) + 1 if iter_numbers else 1
    next_key = f"iter_{next_iter}"

    # Initialize next iteration if not present
    if next_key not in dict_stopw:
        dict_stopw[next_key] = {
            "de_custom_common": [],
            "de_custom_topic_only": [],
            "en_custom_common": [],
            "en_custom_topic_only": []
        }

    # Save updated dictionary
    write_stopw_to_json(dict_stopw, path)

    return dict_stopw, current_iter, next_key


def build_stopword_lists_from_iterations(dict_stopw):
    """
    Build aggregated stopword lists from all iterations.

    Separates stopwords into:
    - topic modeling
    - sentiment analysis
    - German (de) and English (en)

    Parameters
    ----------
    dict_stopw : dict
        Stopword dictionary containing all iterations.

    Returns
    -------
    dict
        Dictionary with merged stopword lists:
        {
            "topic_de": [...],
            "sentiment_de": [...],
            "topic_en": [...],
            "sentiment_en": [...]
        }
    """

    stopw_topic_de = []
    stopw_sentiment_de = []
    stopw_topic_en = []
    stopw_sentiment_en = []

    # Iterate through all stopword iterations
    for iter_key, iteration in dict_stopw.items():

        if iter_key == "iter_0":
            # Base stopwords (NLTK + VADER)
            stopw_topic_de += iteration.get("de_nltk_common", [])
            stopw_sentiment_de += iteration.get("de_nltk_common", [])

            stopw_topic_en += iteration.get("en_nltk_common", [])
            stopw_sentiment_en += iteration.get("en_nltk_common", [])

            # Additional topic-specific sentiment words
            stopw_topic_de += iteration.get("de_vader_topic_only", [])
            stopw_topic_en += iteration.get("en_vader_topic_only", [])

        else:
            # Custom common stopwords (affect both tasks)
            stopw_topic_de += iteration.get("de_custom_common", [])
            stopw_sentiment_de += iteration.get("de_custom_common", [])

            stopw_topic_en += iteration.get("en_custom_common", [])
            stopw_sentiment_en += iteration.get("en_custom_common", [])

            # Topic-specific stopwords
            stopw_topic_de += iteration.get("de_custom_topic_only", [])
            stopw_topic_en += iteration.get("en_custom_topic_only", [])

    # Remove duplicates and sort
    return {
        "topic_de": sorted(set(stopw_topic_de)),
        "sentiment_de": sorted(set(stopw_sentiment_de)),
        "topic_en": sorted(set(stopw_topic_en)),
        "sentiment_en": sorted(set(stopw_sentiment_en)),
    }


def batch_lemmatizer(texts, nlp, stopw, desc, n_process):
    """
    Perform batch lemmatization with filtering using spaCy.

    The function processes texts in batches, removes stopwords,
    filters tokens, and returns cleaned token lists.

    Parameters
    ----------
    texts : iterable
        Collection of input texts.

    nlp : spacy.lang object
        Loaded spaCy language model.

    stopw : set
        Set of stopwords to exclude.

    desc : str
        Description for progress bar display.

    n_process : int
        Number of parallel processes.

    Returns
    -------
    list of list of str
        List of tokenized and lemmatized sentences.
    """

    sentences = []

    # Process texts in parallel batches
    reviews = nlp.pipe(texts, batch_size=200, n_process=n_process)

    for review in tqdm(reviews, total=len(texts), desc=desc):
        lemmas = [
            token.lemma_.lower()
            for token in review
            if token.is_alpha                      # Only alphabetic tokens
            and len(token) > 2                     # Minimum token length
            and not token.is_stop                 # Exclude spaCy stopwords
            and token.lemma_.lower() not in stopw # Exclude custom stopwords
        ]

        if lemmas:
            sentences.append(lemmas)

    return sentences


def clean_and_tokenize(df, stopw_de, stopw_en, n_process):
    """
    Clean and tokenize text data for German and English separately.

    This function:
    - Loads spaCy models
    - Splits dataset by language
    - Applies lemmatization and stopword filtering
    - Builds document representations and vocabularies

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns 'review' and 'language'.

    stopw_de : list
        German stopwords.

    stopw_en : list
        English stopwords.

    n_process : int
        Number of parallel processes.

    Returns
    -------
    dict
        Dictionary with processed data per language:
        {
            "de": {
                "sentences": [...],
                "documents": [...],
                "vocabulary": [...],
                "index": {...}
            },
            "en": { ... }
        }
    """

    # Convert stopword lists to sets for faster lookup
    stopw_de = set(stopw_de)
    stopw_en = set(stopw_en)

    # Load spaCy models for both languages
    nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Split dataset by language
    df_de = df[df["language"] == "de"]["review"].fillna("").astype(str)
    df_en = df[df["language"] == "en"]["review"].fillna("").astype(str)

    result = {}

    # Process both languages
    for lang, texts, nlp, stopw, desc in [
        ("de", df_de, nlp_de, stopw_de, "Processing German"),
        ("en", df_en, nlp_en, stopw_en, "Processing English"),
    ]:

        # Apply batch lemmatization
        sentences = batch_lemmatizer(
            texts=texts,
            nlp=nlp,
            stopw=stopw,
            desc=desc,
            n_process=n_process
        )

        # Convert token lists into space-separated documents
        documents = [" ".join(sentence) for sentence in sentences]

        # Build vocabulary
        vocabulary = sorted(set(word for sent in sentences for word in sent))

        # Map words to indices
        index = {word: i for i, word in enumerate(vocabulary)}

        # Store results per language
        result[lang] = {
            "sentences": sentences,
            "documents": documents,
            "vocabulary": vocabulary,
            "index": index
        }

    return result

def run_preprocessing_loop(
    df_reviews,
    stopword_path,
    dict_stopw_default,
    top_n=30,
    n_process=1,
):
    """
    Runs an interactive preprocessing loop for iterative stopword refinement.

    This function:
    - Loads or initializes a stopword configuration (JSON)
    - Iteratively refines stopwords based on top-token inspection
    - Cleans and tokenizes data separately for topic modeling and sentiment analysis
    - Displays token comparison tables for manual review
    - Allows user-driven iteration via CLI input

    Parameters
    ----------
    df_reviews : pd.DataFrame
        Input dataset containing at least 'review' and 'language' columns.

    stopword_path : str
        Path to the stopword JSON file.

    dict_stopw_default : dict
        Default stopword structure used if JSON file does not exist.

    top_n : int, optional (default=30)
        Number of top tokens to display in comparison.

    n_process : int, optional (default=1)
        Number of parallel processes for spaCy pipeline.

    Returns
    -------
    tuple
        (data_topic, data_sentiment, dict_stopw)

        - data_topic: cleaned/tokenized data for topic modeling
        - data_sentiment: cleaned/tokenized data for sentiment analysis
        - dict_stopw: final stopword dictionary
    """

    while True:
        # Load or initialize stopword JSON
        if os.path.exists(stopword_path) and os.path.getsize(stopword_path) > 0:
            print("Stopword file found -> loading")
            dict_stopw = load_stopw_from_json(stopword_path)

            # Ensure base iteration exists
            if "iter_0" not in dict_stopw:
                dict_stopw["iter_0"] = dict_stopw_default["iter_0"]
                write_stopw_to_json(dict_stopw, stopword_path)

        else:
            print("Stopword file missing or empty -> creating new file")
            dict_stopw = dict_stopw_default
            write_stopw_to_json(dict_stopw, stopword_path)

        # Merge stopwords across all iterations
        stopwords_all = build_stopword_lists_from_iterations(dict_stopw)

        # Extract stopword sets per task/language
        stopw_topic_de = stopwords_all["topic_de"]
        stopw_sent_de = stopwords_all["sentiment_de"]
        stopw_topic_en = stopwords_all["topic_en"]
        stopw_sent_en = stopwords_all["sentiment_en"]

        # Topic preprocessing
        print("\nCleaning Data - Topic Modeling")
        data_topic = clean_and_tokenize(
            df_reviews,
            stopw_de=stopw_topic_de,
            stopw_en=stopw_topic_en,
            n_process=n_process
        )

        # Sentiment preprocessing
        print("\nCleaning Data - Sentiment Analysis")
        data_sentiment = clean_and_tokenize(
            df_reviews,
            stopw_de=stopw_sent_de,
            stopw_en=stopw_sent_en,
            n_process=n_process
        )

        # Token comparison
        print()
        compare_top_tokens(data_topic, data_sentiment, top_n)

        # Get current iteration key
        current_iter = get_current_iteration_key(dict_stopw)

        print(
            f"\nPlease review the top tokens and optionally extend them under "
            f"'{current_iter}' in {stopword_path}."
        )
        print("1 = JSON updated manually, start next iteration")
        print("2 = finished, continue with topic modeling")

        choice = input("Selection: ").strip()

        if choice == "1":
            # Reload JSON after user edits
            dict_stopw = load_stopw_from_json(stopword_path)

            # Add next iteration
            add_iter_to_json(dict_stopw, stopword_path)

        elif choice == "2":
            print("\nFinal stopwords applied. Proceeding to topic modeling.")
            break

        else:
            print("\nInvalid input. Please select 1 or 2.")

    return data_topic, data_sentiment, dict_stopw