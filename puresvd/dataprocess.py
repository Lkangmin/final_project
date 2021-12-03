# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from operator import itemgetter


def deployUninterToTrainSet(trainSet, adjList_user, adjList_item, data, validation, theta, lowRatingValue, childDirectory = False):
    #if data == "ML10M":
    #    uninterFile = "../../../"+data+"/"+validation+".uninter.train"
    #else:
    uninterFile = "datasets/"+data+"/"+validation+".uninter."+theta+".train"
    if childDirectory == True:
        uninterFile = "../"+uninterFile
        
    # read file
    uninterSet = []
    userIndex = 0
    for line in open(uninterFile):
        uninters = line.strip().split(' ')
        uninters = list(map(int, uninters))
        for itemId in uninters:
            trainSet[userIndex].append([itemId, lowRatingValue])
            adjList_user[userIndex].append(itemId)
            adjList_item[itemId].append(userIndex)
        userIndex = userIndex + 1
        
    # sort
    for u, i_list in trainSet.items():
        i_list.sort(key=itemgetter(0))
        
    print("loading uninterSet done.... theta = "+theta)

    
# The training dictionary is used in all the projects as a general purpose
def loadTrainDictionary (data, validation, p, zeroInjectionMode = False, childDirectory = False):
    trainSet = defaultdict(list)
    adjList_user = defaultdict(list)
    adjList_item = defaultdict(list)

    trainFile = "datasets/"+data+"/"+validation+".train"
    if childDirectory == True:
        trainFile = "../"+trainFile
        
    print(trainFile)
    
    max_u_id = -1
    max_i_id = -1
    
    for line in open(trainFile):
        userId, itemId, rating = line.strip().split(' ')
        
        userId = int(userId)
        itemId = int(itemId)
        rating = float(rating)

        adjList_user[userId].append(itemId)
        adjList_item[itemId].append(userId)

        if p == "explicit":
            trainSet[userId].append([itemId, rating])
        else:
            trainSet[userId].append(itemId)

        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
        
    print("loading TrainDictionary done....")

    userCount = max_u_id+1
    itemCount = max_i_id+1
    
    print(userCount)
    print(itemCount)
    
    # sort
    if p == "explicit":
        for u, i_list in trainSet.items():
            i_list.sort(key=itemgetter(0))
    else:
        for u, i_list in trainSet.items():
            i_list.sort()
    
    print("sorting TrainDictionary done....")

    if zeroInjectionMode == True:
        maskingVal = 9999
    else:
        maskingVal = -9999

    testMask = defaultdict(lambda: [0] * (itemCount))
    for userId, i_list in trainSet.items():
        for item in i_list:
            if p == "explicit":
                itemId = item[0]
            else:
                itemId = item
            testMask[userId][itemId] = maskingVal

    testMaskArray = []
    for batchId in range(userCount):
        testMaskArray.append(testMask[batchId])
    testMaskArray = np.array(testMaskArray)

    return userCount, itemCount, trainSet, adjList_user, adjList_item, testMaskArray


# loading ground truth
def loadTestData(data, validation, threshold=3, childDirectory = False):
    testSet = defaultdict(list)
    testFile = "datasets/" + data + "/" + validation + ".test"
    if childDirectory == True:
        testFile = "../"+testFile
        
    print(testFile)

    # read test file
    for line in open(testFile):
        userId, itemId, rating = line.strip().split(' ')
        rating = int(float(rating))
        userId = int(userId)
        itemId = int(itemId)

        if rating > threshold:
            testSet[userId].append(itemId)

    print("loading test set done....")

    return testSet

def getNumOfRatings(adjList_user, adjList_item):
    numOfRatingsPerUser = []
    for userId, i_list in adjList_user.items():
        numOfRatingsPerUser.append(len(i_list))

    numOfRatingsPerItem = []
    for itemId, u_list in adjList_item.items():
        numOfRatingsPerItem.append(len(u_list))

    return numOfRatingsPerUser, numOfRatingsPerItem

# Training data loading for Point-wise methods
def loadTrainRatings (trainSet):

    ratings = []
    mu = 0

    for userId, i_list in trainSet.items():
        for item in i_list:
            itemId, rating = item[0], item[1]
            ratings.append([userId, itemId, float(rating)])
            mu += float(rating)

    mu = mu/len(ratings)
    
    print(len(ratings))
    print(mu)
    print("loading TrainRatings done....")
    
    return ratings, mu

# Training data loading for Vector-wise methods
def loadTrainVectors (trainSet, userCount, itemCount, method, feedback, newUserCount = 0, newItemCount = 0, lowv = 0, conf = 0):

    if method == "userBased":
        batchCount = userCount + newUserCount
        non_batchCount = itemCount + newItemCount
    else:
        batchCount = itemCount + newItemCount
        non_batchCount = userCount + newUserCount


    trainDict = defaultdict(lambda: [0] * non_batchCount)
    trainMask = defaultdict(lambda: [0] * non_batchCount)
   
    for userId, i_list in trainSet.items():
        for item in i_list:
            if feedback == "explicit":
                itemId, rating = item[0], item[1]
            else:
                itemId = item
                rating = 1.0

            if method == "userBased":
                trainDict[userId][itemId] = rating
                trainMask[userId][itemId] = 1.0
            else:
                trainDict[itemId][userId] = rating
                trainMask[itemId][userId] = 1.0

    allDataArray = []
    allMaskArray = []

        
    for batchId in range(batchCount):
        allDataArray.append(trainDict[batchId])
        allMaskArray.append(trainMask[batchId])
    allDataArray = np.array(allDataArray)
    allMaskArray = np.array(allMaskArray)

    print("loading TrainVectors done....")
    
    return allDataArray, allMaskArray
