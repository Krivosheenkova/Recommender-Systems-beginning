# metrics.py
import numpy as np
from typing import List


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1


def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)


def precision(recommended_list, bought_list):
    return len(set(bought_list).intersection(set(recommended_list[:]))) / max(
        len(set(recommended_list)), 1)


def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()


def recall(recommended_list, bought_list):
    return len(set(bought_list).intersection(set(recommended_list[:]))) / max(
        len(set(bought_list)), 1
    )


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = list(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[recommended_list <= k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)


    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant


def dcg_at_k(recommended_list, bought_list, k=5):
    """
  Discounted Cumulative Gain
  ------
    Popular measure for evaluating web search and
        related tasks
    Two assumptions:
       - Highly relevant documents are more useful
        than marginally relevant document
       - The lower the ranked position of a relevant
        document, the less useful it is for the user,
        since it is less likely to be examined
    
    Gain is accumulated starting at the top of the
        ranking and may be reduced, or discounted, at lower ranks.
        Typical discount is 1/log (rank)
    """
    bought_list = np.array(bought_list)
    recommended_list = recommended_list[:k]
    rec_len = len(recommended_list)

    bool_mask = np.isin(recommended_list, bought_list)
    flags = bool_mask * 1

    sum_ = 0
    for i in range(1, rec_len+1):
        sum_ += discount(flags[i-1], i)
    dcg_score = sum_ / rec_len

    return dcg_score


def ndcg(recommended_list, bought_list, k=5):
    """
    Compute Normalized Discounted Cumulative Gain.
    Sum the true scores ranked in the order induced by the predicted scores, 
    after applying a logarithmic discount. Then divide by the best possible score 
    (Ideal DCG, obtained for a perfect ranking) to obtain a score between 0 and 1.

This ranking metric returns a high value if true labels are ranked high by y_score.
    """
    def ideal_dcg(rec_len):
        flags = np.ones(rec_len)
        sum_ = 0
        for i in range(1, rec_len+1):
            sum_ += discount(flags[i-1], i)
        ideal_dcg = sum_ / rec_len
        return ideal_dcg

    dcg_score = dcg_at_k(recommended_list, bought_list, k)

    if k:
        rec_len = k   
    else:
        rec_len = len(recommended_list)
    
    return dcg_score / ideal_dcg(rec_len)



def discount(flag, i):
    if i <= 2:
        return flag / i
    else:
        return flag / np.log2(i)
