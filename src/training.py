# src/training.py

from itertools import product
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

def tune_lda_models(
    data,
    topic_grid,
    alpha_grid,
    eta_grid,
    max_iter=20,
    learning_method="batch",
    random_state=42,
    top_n=10,
    lang=None
):
    tuning_results = []

    X = data["matrix"]
    texts = tokenize_docs(data["documents"])

    total_runs = len(topic_grid) * len(alpha_grid) * len(eta_grid)

    with tqdm(total=total_runs, desc=f"LDA Tuning ({lang})") as pbar:

        for n_topics, alpha, eta in product(topic_grid, alpha_grid, eta_grid):

            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=random_state,
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

            topic_counts = pd.Series(
                topic_matrix.argmax(axis=1)
            ).value_counts(normalize=True)

            perplexity = model.perplexity(X)

            tuning_results.append({
                "n_topics": n_topics,
                "alpha": alpha,
                "eta": eta,
                "coherence": coherence,
                "largest_topic_share": topic_counts.max(),
                "perplexity": perplexity,
                "topic_terms": topic_terms,
                "doc_topic_matrix": topic_matrix
            })

            pbar.update(1)
            pbar.set_postfix({
                "k": n_topics,
                "coh": f"{coherence:.2f}",
                "perp": f"{perplexity:.0f}"
            })

    return pd.DataFrame(tuning_results)


def tune_lsa_models(
    data,
    topic_grid,
    n_iter_grid,          # ← neu
    random_state=42,
    top_n=10,
    lang=None
):
    tuning_results = []

    X = data["matrix"]
    texts = tokenize_docs(data["documents"])

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
                "doc_topic_matrix": topic_matrix
            })

            pbar.update(1)
            pbar.set_postfix({
                "k": n_topics,
                "iter": n_iter,
                "coh": f"{coherence:.2f}",
                "var": f"{explained_var:.2f}"
            })

    return pd.DataFrame(tuning_results)
    
def tokenize_docs(documents):

    return [
        str(doc).lower().split()
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
    
def evaluate_and_score_tuning_generic(
    tuning_df,
    score_metrics,
    score_col="overall_score"
):
    df = tuning_df.copy()
    score = 0

    for metric in score_metrics:
        col = metric["col"]
        weight = metric["weight"]
        higher_is_better = metric["higher_is_better"]

        min_val = df[col].min()
        max_val = df[col].max()

        if max_val == min_val:
            norm = 1
        else:
            norm = (df[col] - min_val) / (max_val - min_val)

        if not higher_is_better:
            norm = 1 - norm

        score += weight * norm

    df[score_col] = score

    return df.sort_values(score_col, ascending=False)