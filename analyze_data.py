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

from src.data_utils import (
    remove_duplicate_texts,
    load_vader_words,
    load_gervader_words,
)

from src.preprocessing import run_preprocessing_loop

from src.training import (
    vectorize,
    tune_lda_models,
    tune_lsa_models,
)

from src.model_utils import (
    build_sentiment_df,
    get_selected_topic_model,
    compute_dominant_topics,
)

from src.evaluation import (
    merge_topic_sentiment,
    build_topic_sentiment_summary,
)

from src.visualization import (
    display_tuning_results,
    display_topic_model_summary,
    plot_topic_words,
    plot_topic_sizes,
    compare_topic_sizes,
    plot_topic_overlap_matrix,
    plot_topic_sentiment,
    show_reviews_for_topic,
    style_topic_sentiment,
)


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

top_n = 30       # Number of top tokens to display
n_process = 1    # Number of parallel processes (performance tuning)


# Run interactive preprocessing pipeline:
# - Loads or initializes stopword configuration (JSON)
# - Applies iterative stopword refinement (user-guided)
# - Cleans and tokenizes text for both topic modeling and sentiment analysis
# - Displays top token comparisons for validation
# - Repeats until user confirms final stopw

data_topic, data_sentiment, dict_stopw = run_preprocessing_loop(
    df_reviews=df_reviews,
    stopword_path=stopword_path,
    dict_stopw_default=dict_stopw_default,
    top_n=top_n,
    n_process=n_process
)



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

# Perform LDA parameter tuning per language using Grid Search.
# Explores multiple configurations to identify optimal topic models
# for each language-specific dataset.

# -----------------------------
# CONFIGURATION
# -----------------------------

# Full parameter grid (production mode - slow but thorough)
lda_config_prod = {
    "topic_grid": [5, 10, 15, 20],   # number of topics to test
    "alpha_grid": [0.01, 0.1],       # document-topic prior (sparsity control)
    "eta_grid": [0.01, 0.1],         # topic-word prior
    "max_iter_grid": [10, 20],       # number of training iterations
    "learning_method": "batch",      # batch learning for stable convergence
}

# Minimal parameter grid (test mode - fast execution)
lda_config_test = {
    "topic_grid": [10],          # single topic value
    "alpha_grid": [0.1],         # single prior
    "eta_grid": [0.1],           # single prior
    "max_iter_grid": [10],       # minimal iterations
    "learning_method": "batch",
}

# -----------------------------
# SWITCH BETWEEN MODES
# -----------------------------

TEST_MODE = False  # set to False for full tuning

lda_config = lda_config_test if TEST_MODE else lda_config_prod


# -----------------------------
# RUN TUNING
# -----------------------------

# Dictionary to store tuning results separately for each language
lda_tuning_results_by_lang = {}

# Iterate over language-specific BOW vectorized datasets
for lang, data in vectors_topic["bow_by_lang"].items():

    # Apply selected configuration (test or production)
    lda_tuning_results_by_lang[lang] = tune_lda_models(
        data=data,
        topic_grid=lda_config["topic_grid"],
        alpha_grid=lda_config["alpha_grid"],
        eta_grid=lda_config["eta_grid"],
        max_iter_grid=lda_config["max_iter_grid"],
        random_states=lda_config["random_states"],
        learning_method=lda_config["learning_method"],
        lang=lang
    )

# +
############
# LDA/BoW - Tuning Validation (en|de)
############

# Define evaluation configuration for LDA tuning results.
# Each metric includes:
# - col: column name in the tuning results
# - weight: contribution to overall score
# - higher_is_better: optimization direction
# - good_quantile / bad_quantile: thresholds for scoring normalization
# - format: display formatting
lda_tuning_config = [
    {"col": "coherence",               "weight": 0.30, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.2f}"},
    {"col": "coherence_std",           "weight": 0.10, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.2f}"},
    {"col": "perplexity",              "weight": 0.20, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    {"col": "largest_topic_share",     "weight": 0.15, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    {"col": "avg_word_overlap",        "weight": 0.10, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
]

# Evaluate and display LDA tuning results per language:
# - Computes weighted scores based on defined metrics
# - Ranks models by overall performance
# - Displays top configurations with formatted styling
lda_tuning_interpreted_by_lang = display_tuning_results(
    results_by_lang=lda_tuning_results_by_lang,
    tuning_config=lda_tuning_config,
    title="LDA/BoW Tuning",
    head_rows=15,  # Limit output to top-performing configurations
    hide_cols=[
        # Hide non-relevant or heavy columns for display clarity
        "topic_words", "topic_terms", "doc_topic_matrix", "model",
        "max_word_overlap", "largest_topic_share_std", "perplexity_std"
    ]
)


# +
############
# LSA/TF-IDF - Model Tuning (en|de)
############

# Perform parameter tuning for LSA (Latent Semantic Analysis)
# on TF-IDF features, evaluated separately for each language.
# The goal is to identify optimal semantic dimensionality and
# stable decomposition settings.


# Dictionary to store tuning results per language (e.g. 'en', 'de')
lsa_tuning_results_by_lang = {}

# Iterate over language-specific TF-IDF vectorized datasets
for lang, data in vectors_topic["tfidf_by_lang"].items():

    # Perform LSA model tuning using different hyperparameter combinations:
    # - topic_grid: number of latent semantic components (topics)
    #   higher values → finer topic separation, but harder interpretation
    # - n_iter_grid: number of SVD iterations
    #   more iterations → better convergence and stability, but slower runtime
    # - random_state: fixed seed for reproducibility
    # - lang: used for logging or progress tracking
    lsa_tuning_results_by_lang[lang] = tune_lsa_models(
        data=data,
        topic_grid=[10, 20, 30],
        n_iter_grid=[10, 20, 30],
        random_state=42,
        lang=lang
    )

# +
############
# LSA/TF-IDF - Tuning Validation (en|de)
############

# Define evaluation configuration for LSA tuning results.
# Each metric contributes to the final score using a weighted scheme:
# - col: metric name in the tuning results DataFrame
# - weight: importance of the metric in overall ranking
# - higher_is_better: optimization direction
# - good_quantile / bad_quantile: thresholds used for normalization/scoring
# - format: display formatting for readability
lsa_tuning_config = [
    {"col": "avg_word_overlap",     "weight": 0.16, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
    {"col": "coherence",            "weight": 0.42, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "explained_variance",   "weight": 0.17, "higher_is_better": True,  "good_quantile": 0.75, "bad_quantile": 0.25, "format": "{:.4f}"},
    {"col": "largest_topic_share",  "weight": 0.25, "higher_is_better": False, "good_quantile": 0.25, "bad_quantile": 0.75, "format": "{:.4f}"},
]

# Evaluate and display LSA tuning results for each language:
# - Computes normalized and weighted scores based on the configuration
# - Ranks model configurations by overall performance
# - Displays the top results with formatted styling for comparison
lsa_tuning_interpreted_by_lang = display_tuning_results(
    results_by_lang=lsa_tuning_results_by_lang,
    tuning_config=lsa_tuning_config,
    title="LSA/TF-IDF Tuning",
    head_rows=10,  # Limit output to top-performing configurations
    hide_cols=[
        # Hide large or non-essential columns to improve readability
        "topic_words", "topic_terms", "doc_topic_matrix", "model"
    ]
)

# +
############
# Result Validation/ Model Selection
############

### Top Word Comparison of the Top X Models (EN | DE)
# Displays the most important words for each topic.
# Purpose:
# - Are the topics semantically coherent and interpretable?
# - Are there generic or poorly defined topics?
# - Are the topics redundant or clearly distinct?
display_topic_model_summary(
    tuning_interpreted_by_lang=lsa_tuning_interpreted_by_lang,
    model_name="LSA/TF-IDF",
    top_k=6
)

display_topic_model_summary(
    tuning_interpreted_by_lang=lda_tuning_interpreted_by_lang,
    model_name="LDA/BoW",
    top_k=6
)


# +
### Topic Visualization of the Selected Model
# Displays the keyword distribution of a selected model.
# Purpose:
# - Validate the qualitative topic structure.
# - Assess the balance of document assignments across topics.
# - Identify dominant, underrepresented, or empty topics.

plot_topic_words(
    lsa_tuning_interpreted_by_lang["de"].iloc[:5]
)

# +
### Topic Size Comparison of Top Models (Table)
# Displays the number of documents assigned to each topic
# for all candidate models.
# Purpose:
# - Assess the balance of topic distributions.
# - Identify dominant or underrepresented topics.
# - Detect empty or nearly empty topics.
# - Support qualitative model selection.
    
display(compare_topic_sizes(
    lsa_tuning_interpreted_by_lang["en"],
    method="lsa",
    title="=== LSA Topic-Größen (EN) ==="
))

# +
### Topic Size Comparison of Top X or Selected Models (Visual)

for i in range(5):
    row = lsa_tuning_interpreted_by_lang["en"].iloc[i]
        
    plot_topic_sizes(
        doc_topic_matrix=row["doc_topic_matrix"],
        method="lsa",
        title=f"LSA Model {i}"
    )
    
#plot_topic_sizes(
#    doc_topic_matrix=best_row["doc_topic_matrix"],
#    method="lsa",
#    title="LSA Topic Sizes"
#)

for i in range(5):
    row = lda_tuning_interpreted_by_lang["de"].iloc[i]
        
    plot_topic_sizes(
        doc_topic_matrix=row["doc_topic_matrix"],
        method="lda",
        title=f"LDA Model {i}"
    )
#plot_topic_sizes(
#    doc_topic_matrix=best_row["doc_topic_matrix"],
#    method="lda",
#    title="LDA Topic Sizes"
#)

# +
# Displays the overlap of top keywords between topics.
# Purpose:
# - Identify redundant or highly similar topics.
# - Assess whether the number of topics is too high.
# - Evaluate the degree of topic separation.

plot_topic_overlap_matrix(
    lsa_tuning_interpreted_by_lang["en"]
)

# +
############
# FINAL TOPIC + SENTIMENT PIPELINE
# Separate Vector Spaces:
# - Topic Modeling: captures semantic structure of documents
# - Sentiment Analysis: captures emotional polarity independently
############

# 1. Manually Select Final Models
# Select best-performing configurations based on prior evaluation.
# Each selection defines:
# - modeling approach (LDA or LSA)
# - chosen parameter configuration (row)
# - corresponding vector spaces for topic and sentiment analysis
selected_models = {

    "de": {
        "method": "lda",
        "title": "LDA BoW (de)",

        # Manually selected model (not necessarily top-ranked)
        "row": lda_tuning_interpreted_by_lang["de"].iloc[0],

        # Use Bag-of-Words representation for both tasks
        "vector_topic": vectors_topic["bow_by_lang"]["de"],
        "vector_sentiment": vectors_sentiment["bow_by_lang"]["de"],
    },

    "en": {
        "method": "lsa",
        "title": "LSA TFIDF (en)",

        # Manually selected model (not necessarily top-ranked)
        "row": lsa_tuning_interpreted_by_lang["en"].iloc[0],

        # Use TF-IDF representation for both tasks
        "vector_topic": vectors_topic["tfidf_by_lang"]["en"],
        "vector_sentiment": vectors_sentiment["tfidf_by_lang"]["en"],
    }
}

# 2. Extract Final Topic Models
# Retrieve model components (topic terms, document-topic matrix, features)
# for the selected configurations
fitted_models = {}

for lang, selection in selected_models.items():
    fitted_models[lang] = get_selected_topic_model(selection)

# 3. Determine Dominant Topic per Document
# For each document:
# - Identify the most relevant (dominant) topic
# - Compute topic strength
doc_topics = {}

for lang, selection in selected_models.items():

    result = get_selected_topic_model(selection)

    doc_topics[lang] = compute_dominant_topics(
        doc_topic_matrix=result["doc_topic_matrix"],
        documents=result["documents"],
        method=selection["method"]
    )

# 4. Sentiment Analysis
# Compute sentiment score and label (positive / neutral / negative)
# for each document, independent of topic modeling
sentiments = {}

for lang, selection in selected_models.items():

    sentiments[lang] = build_sentiment_df(
        selection["vector_sentiment"],
        lang=lang
    )

# 5. Merge Topic + Sentiment
# Combine topic assignments with sentiment results
# to associate each document with both semantic and emotional information
doc_topics_sentiment = {}

for lang in doc_topics.keys():

    doc_topics_sentiment[lang] = merge_topic_sentiment(
        doc_topics[lang],
        sentiments[lang]
    )

# 6. Aggregate Topic + Sentiment Matrix
# Compute aggregated statistics per topic:
# - number of documents
# - average topic strength
# - average sentiment score
# - distribution of sentiment labels
topic_sentiment_summary = {}

for lang, df in doc_topics_sentiment.items():

    topic_sentiment_summary[lang] = build_topic_sentiment_summary(df)


# 7. Display Final Topic + Sentiment Matrices

for lang, summary_df in topic_sentiment_summary.items():

    selection = selected_models[lang]
    topic_terms = selection["row"]["topic_terms"]

    summary_display = summary_df.copy()

    # Top words per topic
    topic_keywords = {
        int(topic_name.split()[-1]) - 1: ", ".join(
            term["word"] for term in terms
        )
        for topic_name, terms in topic_terms.items()
    }

    summary_display["top_words"] = (
        summary_display["dominant_topic"]
        .map(topic_keywords)
    )

    print(f"\n=== Topic + Sentiment Vergleich ({lang}) ===")

    display(
        style_topic_sentiment(
            summary_display,
            f"{selection['title']} - Topic & Sentiment Matrix",
            pos_threshold=0.05,
            neg_threshold=-0.05
        )
    )

# +
# Displays the average sentiment score for each topic.
# Purpose:
# - Identify topics with predominantly positive or negative sentiment.
# - Detect emotionally polarized topics.
# - Compare topics in terms of their semantic content and sentiment.

plot_topic_sentiment(
    topic_sentiment_summary["en"],
    title="EN Topic Sentiment"
)

plot_topic_sentiment(
    topic_sentiment_summary["de"],
    title="DE Topic Sentiment"
)
# -

# Displays representative documents for a selected topic and sentiment class.
# Purpose:
# - Validate the semantic coherence of the discovered topic.
# - Assess whether the assigned documents match the topic keywords.
# - Verify whether the sentiment classification is consistent with the document content.
# - Support the qualitative evaluation of the topic model.
pd.set_option("display.max_colwidth", 1000)
show_reviews_for_topic(
    doc_topics_sentiment,
    lang="en",
    topic_id=1,
    sentiment_label="negative",
    n=10
)






