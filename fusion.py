import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict


def combsum_fusion(ranked_lists):
    """
    Perform CombSum fusion on multiple ranked lists with efficient score normalization.

    :param ranked_lists: A list of lists, where each inner list contains tuples of (doc_id, score)
    :return: A list of tuples (doc_id, fused_score) sorted by fused_score in descending order
    """
    # Create a set of all unique document IDs
    all_doc_ids = set(
        doc_id for ranked_list in ranked_lists for doc_id, _ in ranked_list)

    # Create a dictionary to map document IDs to their index in the numpy array
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}

    # Initialize a numpy array to hold all scores
    scores_array = np.zeros((len(all_doc_ids), len(ranked_lists)))

    # Fill the scores array
    for list_idx, ranked_list in enumerate(ranked_lists):
        for doc_id, score in ranked_list:
            scores_array[doc_id_to_index[doc_id], list_idx] = score

    # Normalize scores using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores_array)

    # Sum up the normalized scores
    fused_scores = np.sum(normalized_scores, axis=1)

    # Create the fused list of (doc_id, score) tuples
    fused_list = [(doc_id, fused_scores[idx])
                  for doc_id, idx in doc_id_to_index.items()]

    # Sort the fused list by score in descending order
    fused_list.sort(key=lambda x: x[1], reverse=True)

    return fused_list


def rrf_fusion(ranked_lists, k=60):
    """
    Perform Reciprocal Rank Fusion on multiple ranked lists.

    :param ranked_lists: A list of lists, where each inner list contains tuples of (doc_id, score)
    :param k: The constant in the RRF formula (default is 60 as per the original paper)
    :return: A list of tuples (doc_id, fused_score) sorted by fused_score in descending order
    """
    fused_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1 / (k + rank)

    # Sort the fused list by score in descending order
    fused_list = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return fused_list
