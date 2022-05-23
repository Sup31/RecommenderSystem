
from collections import defaultdict
 
def RecommendationsOutput(predictions, customer):

    top_10 = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_10[uid].append((iid, est))

    # Sorting predictions and retrieving top 10
    for uid, user_ratings in top_10.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_10[uid] = user_ratings[:10 ]

    return top_10
