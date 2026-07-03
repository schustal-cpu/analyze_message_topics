# src/model_utils.py

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product, combinations

from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GerVADER.vaderSentimentGER import SentimentIntensityAnalyzer as GerSentimentIntensityAnalyzer

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

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
    
def tune_lda_models(
    data,
    topic_grid,
    alpha_grid,
    eta_grid,
    max_iter_grid=(20,),
    random_states=(42,),
    learning_method="batch",
    top_n=10,
    lang=None
):
    tuning_results = []

    X = data["matrix"]
    texts = tokenize_docs_with_vectorizer(
        data["documents"],
        data["vectorizer"]
    )

    total_runs = (
        len(topic_grid)
        * len(alpha_grid)
        * len(eta_grid)
        * len(max_iter_grid)
    )

    with tqdm(total=total_runs, desc=f"LDA Tuning ({lang})") as pbar:

        for n_topics, alpha, eta, max_iter in product(
            topic_grid,
            alpha_grid,
            eta_grid,
            max_iter_grid
        ):

            coherence_scores = []
            perplexity_scores = []
            largest_topic_share_scores = []
            avg_word_overlap_scores = []
            max_word_overlap_scores = []

            best_seed_result = None
            best_seed_coherence = -np.inf

            for seed in random_states:

                model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=seed,
                    learning_method=learning_method,
                    max_iter=max_iter,
                    doc_topic_prior=alpha,
                    topic_word_prior=eta,
                    evaluate_every=-1
                )

                topic_matrix = model.fit_transform(X)

                topic_terms = get_topic_terms_from_model(
                    model=model,
                    feature_names=data["feature_names"],
                    method_type="lda",
                    top_n=top_n
                )

                topics = [
                    [item["word"] for item in terms]
                    for terms in topic_terms.values()
                ]

                coherence = compute_pmi_coherence(
                    topics=topics,
                    texts=texts
                )

                overlap_results = compute_topic_word_overlap(topics)

                topic_counts = pd.Series(
                    topic_matrix.argmax(axis=1)
                ).value_counts(normalize=True)

                perplexity = model.perplexity(X)

                coherence_scores.append(coherence)
                perplexity_scores.append(perplexity)
                largest_topic_share_scores.append(topic_counts.max())
                avg_word_overlap_scores.append(overlap_results["avg_word_overlap"])
                max_word_overlap_scores.append(overlap_results["max_word_overlap"])

                if coherence > best_seed_coherence:
                    best_seed_coherence = coherence
                    best_seed_result = {
                        "model": model,
                        "topic_terms": topic_terms,
                        "doc_topic_matrix": topic_matrix,
                        "seed": seed
                    }

            tuning_results.append({
                "n_topics": n_topics,
                "alpha": alpha,
                "eta": eta,
                "max_iter": max_iter,

                "coherence": np.mean(coherence_scores),
                "coherence_std": np.std(coherence_scores),

                "perplexity": np.mean(perplexity_scores),
                "perplexity_std": np.std(perplexity_scores),

                "largest_topic_share": np.mean(largest_topic_share_scores),
                "largest_topic_share_std": np.std(largest_topic_share_scores),

                "avg_word_overlap": np.mean(avg_word_overlap_scores),
                "max_word_overlap": np.mean(max_word_overlap_scores),

                "best_seed": best_seed_result["seed"],
                "topic_terms": best_seed_result["topic_terms"],
                "doc_topic_matrix": best_seed_result["doc_topic_matrix"],
                "model": best_seed_result["model"]
            })

            pbar.update(1)
            pbar.set_postfix({
                "k": n_topics,
                "iter": max_iter,
                "coh": f"{np.mean(coherence_scores):.2f}",
                "std": f"{np.std(coherence_scores):.3f}",
                "perp": f"{np.mean(perplexity_scores):.0f}"
            })

    return pd.DataFrame(tuning_results)


def tune_lsa_models(
    data,
    topic_grid,
    n_iter_grid,          # ← neu
    random_state,
    top_n=10,
    lang=None
):
    tuning_results = []

    X = data["matrix"]
    texts = tokenize_docs_with_vectorizer(
        data["documents"],
        data["vectorizer"]
    )

    total_runs = len(topic_grid) * len(n_iter_grid)

    with tqdm(total=total_runs, desc=f"LSA Tuning ({lang})") as pbar:

        for n_topics, n_iter in product(topic_grid, n_iter_grid):

            model = TruncatedSVD(
                n_components=n_topics,
                n_iter=n_iter,
                random_state=random_state
            )

            topic_matrix = model.fit_transform(X)
            
            topic_terms = get_topic_terms_from_model(
                model=model,
                feature_names=data["feature_names"],
                method_type="lsa",
                top_n=top_n
            )
            
            topics = [
                [item["word"] for item in terms]
                for terms in topic_terms.values()
            ]
            
            coherence = compute_pmi_coherence(
                topics=topics,
                texts=texts
            )
            
            overlap_results = compute_topic_word_overlap(topics)

            topic_counts = pd.Series(
                np.abs(topic_matrix).argmax(axis=1)
            ).value_counts(normalize=True)

            explained_var = model.explained_variance_ratio_.sum()
            
            tuning_results.append({
                "n_topics": n_topics,
                "n_iter": n_iter,
                "coherence": coherence,
                "largest_topic_share": topic_counts.max(),
                "explained_variance": explained_var,
                "topic_words": {
                    f"Topic {i + 1}": words
                    for i, words in enumerate(topics)
                },
                "topic_terms": topic_terms,
                "doc_topic_matrix": topic_matrix,
                "avg_word_overlap": overlap_results["avg_word_overlap"],
                "max_word_overlap": overlap_results["max_word_overlap"],
                "model": model
            })

            pbar.update(1)
            pbar.set_postfix({
                "k": n_topics,
                "iter": n_iter,
                "coh": f"{coherence:.2f}",
                "var": f"{explained_var:.2f}"
            })

    return pd.DataFrame(tuning_results)
    
    
def tokenize_docs_with_vectorizer(documents, vectorizer):
    analyzer = vectorizer.build_analyzer()

    return [
        analyzer(str(doc))
        for doc in documents
        if isinstance(doc, str) and doc.strip()
    ]


def compute_doc_freq(texts):
    df = Counter()

    for doc in texts:
        for word in set(doc):
            df[word] += 1

    return df, len(texts)


def compute_co_occurrence(texts):
    co_occ = Counter()

    for doc in texts:
        unique_words = list(set(doc))

        for i in range(len(unique_words)):
            for j in range(i + 1, len(unique_words)):
                pair = tuple(sorted((unique_words[i], unique_words[j])))
                co_occ[pair] += 1
    
    return co_occ

def compute_pmi_coherence(topics, texts):
        
    df, n_docs = compute_doc_freq(texts)
    co_occ = compute_co_occurrence(texts)

    topic_scores = []
    
    for topic in topics:
        pair_scores = []

        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                w1 = topic[i]
                w2 = topic[j]

                pair = tuple(sorted((w1, w2)))

                co = co_occ.get(pair, 0)
                df_w1 = df.get(w1, 1)
                df_w2 = df.get(w2, 1)

                # geglättete PMI
                pmi = np.log((co + 1) * n_docs / (df_w1 * df_w2))
                pair_scores.append(pmi)

        if pair_scores:
            topic_scores.append(np.mean(pair_scores))

    return np.mean(topic_scores) if topic_scores else np.nan

def compute_topic_word_overlap(topics):
    similarities = []

    for topic_a, topic_b in combinations(topics, 2):
        set_a = set(topic_a)
        set_b = set(topic_b)

        union = set_a | set_b
        intersection = set_a & set_b

        if len(union) == 0:
            continue

        similarities.append(len(intersection) / len(union))

    if not similarities:
        return {
            "avg_word_overlap": 0.0,
            "max_word_overlap": 0.0
        }

    return {
        "avg_word_overlap": float(np.mean(similarities)),
        "max_word_overlap": float(np.max(similarities))
    }

def get_topic_terms_from_model(model, feature_names, method_type, top_n=10):
    topic_terms = {}

    for topic_idx, topic_weights in enumerate(model.components_, start=1):

        if method_type == "lda":
            weights_for_ranking = topic_weights
        elif method_type == "lsa":
            weights_for_ranking = np.abs(topic_weights)
        else:
            raise ValueError("method_type muss 'lda' oder 'lsa' sein")

        top_indices = weights_for_ranking.argsort()[:-top_n - 1:-1]
        top_weights = weights_for_ranking[top_indices]

        weight_sum = top_weights.sum()
        top_weights_norm = top_weights / weight_sum if weight_sum != 0 else top_weights

        topic_terms[f"Topic {topic_idx}"] = [
            {
                "word": feature_names[i],
                "weight": float(topic_weights[i]),
                "abs_weight": float(weights_for_ranking[i]),
                "weight_pct": float(w * 100),
            }
            for i, w in zip(top_indices, top_weights_norm)
        ]

    return topic_terms


def merge_topic_sentiment(topic_df, sentiment_df):
    """
    Merge topic assignments with sentiment results.

    Parameters
    ----------
    topic_df : pd.DataFrame
        Contains document topics.

    sentiment_df : pd.DataFrame
        Contains sentiment scores and labels.

    Returns
    -------
    pd.DataFrame
        Combined dataset with topic + sentiment information.
    """

    return topic_df.merge(
        sentiment_df,
        on="doc_id",
        how="left",
        suffixes=("_topic", "_sentiment")
    )


def build_topic_sentiment_summary(df):
    """
    Aggregate topic and sentiment data.

    Computes:
    - number of documents per topic
    - average topic strength
    - average sentiment score
    - sentiment distribution (positive / neutral / negative)

    Parameters
    ----------
    df : pd.DataFrame
        Combined topic + sentiment DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated summary per topic.
    """

    return (
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

def get_selected_topic_model(selection):
    """
    Extract the relevant components of a selected topic model.

    Parameters
    ----------
    selection : dict
        Dictionary containing:
        - row: selected tuning result (DataFrame row)
        - vector_topic: vectorized dataset

    Returns
    -------
    dict
        Contains model, topic terms, document-topic matrix,
        and related metadata.
    """

    row = selection["row"]
    data = selection["vector_topic"]

    return {
        "model": row["model"],
        "topic_terms": row["topic_terms"],
        "doc_topic_matrix": row["doc_topic_matrix"],
        "documents": data["documents"],
        "features": data["feature_names"],
    }


def compute_dominant_topics(doc_topic_matrix, documents, method):
    """
    Determine the dominant topic for each document.

    For LDA:
        - Uses probability distribution directly
    For LSA:
        - Uses absolute values due to signed components

    Parameters
    ----------
    doc_topic_matrix : np.ndarray
        Document-topic matrix.

    documents : list of str
        Original documents.

    method : str
        Either "lda" or "lsa".

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - document ID
        - document text
        - dominant topic
        - topic strength
    """

    if method == "lda":
        dominant_topics = doc_topic_matrix.argmax(axis=1)
        topic_strengths = doc_topic_matrix.max(axis=1)

    elif method == "lsa":
        abs_matrix = np.abs(doc_topic_matrix)
        dominant_topics = abs_matrix.argmax(axis=1)
        topic_strengths = abs_matrix.max(axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.DataFrame({
        "doc_id": range(len(documents)),
        "document": documents,
        "dominant_topic": dominant_topics + 1,  # 1-based indexing
        "topic_strength": topic_strengths
    })