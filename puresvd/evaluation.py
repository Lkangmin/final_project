# -*- coding: utf-8 -*-
import numpy as np
import math

def computeTopNAccuracy (testSet, userList_Test, pred_indices, topN, Precision, Recall, NDCG, MRR):
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for user in userList_Test:  # for a user,
            if len(testSet[user]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(testSet[user])
                ndcg = 0

                for j in range(topN[index]):
                    if pred_indices[user][j] in testSet[user]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(testSet[user])
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        
        Precision[index] = sumForPrecision / len(userList_Test)
        Recall[index] = sumForRecall / len(userList_Test)
        NDCG[index] = sumForNdcg / len(userList_Test)
        MRR[index] = sumForMRR / len(userList_Test)

    return Precision, Recall, NDCG, MRR



def prepareMetrics(topN, be):
    Precision = np.zeros(len(topN))
    Recall = np.zeros(len(topN))
    NDCG = np.zeros(len(topN))
    MRR = np.zeros(len(topN))
    
    return Precision, Recall, NDCG, MRR