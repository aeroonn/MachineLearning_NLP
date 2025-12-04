import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import itertools
from math import log

class TopicModelWrapper:
    def __init__(self, topics):
        self.topics = topics

    def get_topics(self):
        return self.topics

def evaluate_bertopic_pmi(
    topic_model,
    docs,
    top_k_coherence: int = 5,
    top_k_diversity: int = 10,
    skip_outlier: bool = True,
    tag: str = "Baseline",
):

    # 1) Extract topic -> list of topic words from BERTopic
    raw_topics = topic_model.get_topics()

    new_keywords = {
        topic_id: [word for word, _ in word_scores]
        for topic_id, word_scores in raw_topics.items()
    }

    # optional: drop outlier topic -1 for evaluation
    if skip_outlier and -1 in new_keywords:
        new_keywords = {tid: kws for tid, kws in new_keywords.items() if tid != -1}

    # 2) Build co-occurrence stats over docs (only topic words)
    # 2.1 Collect all unique keywords from topics
    all_keywords = sorted({w for kws in new_keywords.values() for w in kws})

    vectorizer = CountVectorizer(vocabulary=all_keywords, lowercase=True)
    X = vectorizer.fit_transform(docs)  # shape: (n_docs, n_terms)

    n_docs, n_terms = X.shape

    X_bin = (X > 0).astype(int)

    word_doc_counts = np.asarray(X_bin.sum(axis=0)).ravel()
    p_w = word_doc_counts / n_docs

    cooc_counts = (X_bin.T @ X_bin).toarray()
    p_ij = cooc_counts / n_docs

    vocab = np.array(vectorizer.get_feature_names_out())
    word2id = {w: i for i, w in enumerate(vocab)}

    # 3) Define NPMI helpers and compute topic coherence
    def npmi_pair(w1, w2):
        i = word2id.get(w1)
        j = word2id.get(w2)
        if i is None or j is None:
            return None

        pij = p_ij[i, j]
        if pij == 0:
            return None  # never co-occur

        pi = p_w[i]
        pj = p_w[j]

        pmi = log(pij / (pi * pj))
        return pmi / (-log(pij))

    def topic_npmi_coherence(topic_words, top_k=None):
        if top_k is not None:
            topic_words = topic_words[:top_k]

        scores = []
        for w1, w2 in itertools.combinations(topic_words, 2):
            score = npmi_pair(w1, w2)
            if score is not None:
                scores.append(score)

        if not scores:
            return float("nan")
        return float(np.mean(scores))

    topic_scores = {
        topic_id: topic_npmi_coherence(words, top_k=top_k_coherence)
        for topic_id, words in new_keywords.items()
    }

    coherence_df = pd.DataFrame(
        {
            "Topic": list(topic_scores.keys()),
            "NPMI": list(topic_scores.values()),
        }
    ).sort_values("Topic")

    mean_npmi = float(np.nanmean(coherence_df["NPMI"]))

    # 4) Topic diversity
    def topic_diversity(topics_dict, k=10):
        # collect top-k words for each topic
        topk_words = []
        for tid, words in topics_dict.items():
            topk_words.extend(words[:k])

        if not topk_words:
            return float("nan")

        unique_words = set(topk_words)
        T = len(topics_dict)
        total_words = T * k

        return len(unique_words) / total_words

    diversity = float(topic_diversity(new_keywords, k=top_k_diversity))

    print(f"{tag} Model - NPMI: {mean_npmi:.4f}")
    print(f"{tag} Model - Diversity: {diversity:.4f}")

    return coherence_df, mean_npmi, diversity
