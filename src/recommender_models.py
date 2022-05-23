from collections import defaultdict
from surprise import SVD, BaselineOnly, SlopeOne
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
from top_10_recommendations import RecommendationsOutput
import pandas as pd
import numpy as np

def main():
    #Process Data
    df = pd.read_json('kindle_reviews.json',lines= True)
    #dataframe = pd.read_json('Books_5.json',lines= True)
 
    #Reduce number of columns to minimum required
    dataframe = df[["overall", "reviewerID", "asin"]]
    

    print("Reviews:",dataframe.shape[0])
    print("Users: ", len(np.unique(dataframe.reviewerID)))
    print("Unique Books:", len(np.unique(dataframe.asin)))
    
    #Split the data into testing 20 % and training 80%
    trainingdata = df.sample(frac=0.8)
    testingdata = dataframe.drop(trainingdata.index)
    reader = Reader(rating_scale=(1,5))
    trainset = Dataset.load_from_df(trainingdata[['reviewerID','asin','overall']], reader).build_full_trainset()
    testset = Dataset.load_from_df(testingdata[['reviewerID','asin','overall']], reader).build_full_trainset().build_testset()

    #SVD
    SVDAlgo = SVD(verbose=True, n_epochs=10)
    SVDAlgo.fit(trainset)
    SVDPred= SVDAlgo.test(testset)
    print("\nSVD metrics")
    print(accuracy.mae(SVDPred, verbose=True))
    print(accuracy.rmse(SVDPred, verbose=True))

    #SlopeOne
    SlopeOneAlgo = SlopeOne()
    SlopeOneAlgo.fit(trainset)
    SlopeOnePred = SlopeOneAlgo.test(testset)
    print("SlopeOne metrics")
    print(accuracy.mae(SlopeOnePred, verbose=True))
    print(accuracy.rmse(SlopeOnePred, verbose=True))

    KNNWithMean
    KNN = KNNWithMeans(k=10, sim_options={'name': 'pearson_baseline', 'user_based': False})
    KNN.fit(trainset)
    KNNPred = KNN.test(testset)
    print("KNNWithMean")
    accuracy.rmse(KNNPred, verbose=True)
    accuracy.mae(KNNPred, verbose=True)
    
    advanceMetrics(SVDPred)    
    advanceMetrics(SlopeOnePred)
    advanceMetrics(KNNPred)

    RecommendationsOutput(SVDPred, "A3SPTOKDG7WBLN")

def advanceMetrics(predictions):
    groupedpredictions = defaultdict(list)
    for uid, iid, r_ui, est, details in predictions:
        groupedpredictions[uid].append((est, r_ui))

    precision = dict()
    recall = dict()
    
    for uid, uratings in groupedpredictions.items():
       TotalRelevant = 0
       Recommended = 0
       RelevantRecommended = 0
       #Sort in descending order
       uratings.sort(key=lambda x: x[0], reverse=True)

       #True Positives and False Negatives
       for est, r_ui in uratings:
            if (r_ui > 3):
                TotalRelevant+=1
        
       for est, r_ui in uratings[:10]:
        #FalsePositive
            if (est >= 3):
                Recommended+=1
        #True Positives
            if (est >= 3) and (r_ui >= 3):
                RelevantRecommended+=1
       
       if Recommended != 0:
            precision[uid] = RelevantRecommended / Recommended
       else:
            precision[uid] = 0
       if TotalRelevant != 0:
           recall[uid] = RelevantRecommended / TotalRelevant
       else:
           recall[uid] = 0
    PrecisionSum = 0
    RecallSum = 0
    conversionR = 0
    for precision in precision.values():
       PrecisionSum+=precision
    for recall in recall.values():
        RecallSum+=recall
    totalPrecision = PrecisionSum/len(precision)
    totalRecall = RecallSum/len(recall)
    print("Precision:",totalPrecision)
    print("Recall:",totalRecall)
    print("F-Measure:",(2*totalRecall*totalPrecision)/(totalPrecision+totalRecall))
    for precision in precision.values():
        if precision>0:
            conversionR+=1
    print("Conversion Rate:",conversionR/len(precision.values()))

if __name__ == "__main__":
    main()
