# pip install sparsesvd
#usage:   python puresvd.py

import tensorflow as tf
import numpy as np
import random
import sys
import time
import evaluation
import dataprocess
import utilities
from collections import OrderedDict

from scipy import sparse
from sparsesvd import sparsesvd

topN = [5, 10, 15, 20]

d = "d1"
p = "explicit"

benchmark = ["ML1M", "ML100k", "ciao", "watcha"]
hiddenDim = [5, 10, 15, 20]
hyperParams = OrderedDict()

with tf.Graph().as_default():
    # define top-K
    prediction_top_k = tf.placeholder(tf.float32, [None, None])
    scale_top_k = tf.placeholder(tf.int32)
    top_k = tf.nn.top_k(prediction_top_k, scale_top_k)

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for be in benchmark:
        userCount, itemCount, trainSet, adjList_user, adjList_item, testMaskArray = dataprocess.loadTrainDictionary(be, d, p)
        testSet = dataprocess.loadTestData(be, d)

        userList_test = list(testSet.keys())
        userList_train = list(trainSet.keys())
        batchSize_test = 500

        for h in hiddenDim:
            Precision, Recall, NDCG, MRR = evaluation.prepareMetrics(topN, be)

            # prepare sparse matrix of training data
            sparse_rating_matrix = sparse.lil_matrix((userCount, itemCount))
            for u, i_list in trainSet.items():
                for j in i_list:
                    if p == "explicit":
                        sparse_rating_matrix[u, j[0]] = j[1]
                    else:
                        sparse_rating_matrix[u, j] = 1.0

            P, Sigma, Q = sparsesvd(sparse.csc_matrix(sparse_rating_matrix), h)

            predictedIndices = []

            numOfMinibatches_validation = int(len(userList_train) / batchSize_test) + 1
            numOfLastMinibatch_validation = len(userList_train) % batchSize_test

            t1 = time.time()
            for batchId in range(numOfMinibatches_validation):
                batchUserIndex = []
                start = batchId * batchSize_test
                if batchId == numOfMinibatches_validation-1:   #if it is the last minibatch
                    numOfBatches = numOfLastMinibatch_validation
                else:
                    numOfBatches = batchSize_test
                end = start + numOfBatches

                batchMask = []
                _prediction = []

                for i in range(start, end):
                    # this user
                    userId = userList_train[i]

                    # prediction of this user
                    R = np.dot(P.T[userId], np.dot(np.diag(Sigma), Q))
                    _prediction.append(R)

                _prediction = np.array(_prediction)
                _prediction = _prediction + testMaskArray[start: end]

                _, predict_minibatch = sess.run(top_k, feed_dict={prediction_top_k: _prediction, scale_top_k: topN[-1]})

                predictedIndices.extend(predict_minibatch)

            Precision, Recall, NDCG, MRR = evaluation.computeTopNAccuracy (testSet, userList_test, predictedIndices, topN, Precision, Recall, NDCG, MRR)
            t2 = time.time()
            hyperParams['hiddenDim'] = h

            # report results on files
            utilities.printBaselineResults(be, topN, "pureSVD", [hyperParams], Precision, Recall, NDCG, MRR, str(t2-t1))

    sess.close()
