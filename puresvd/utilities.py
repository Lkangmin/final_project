# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
import linecache
import os
import numpy as np

def getSimilarity(adjList, trainVector, A, B):
    # get the intersection of the two non-zero parts
    interSection = list(set(adjList[A]).intersection(adjList[B]))

    a = trainVector[A][interSection]
    b = trainVector[B][interSection]

    return np.corrcoef(a, b)[0, 1]

def printBaselineResults(be, topN, method, hyperParamsList, Precision, Recall, NDCG, MRR, time):
    for I in range(len(topN)):
        outputFile = be + "_" + method + "_top" + str(topN[I]) + ".csv"
        f = open(outputFile, 'a')

        resultLine = method
        for hyperParams  in hyperParamsList:  
            for values in hyperParams.values():
                resultLine = resultLine + ',' + str(values)

        resultLine = resultLine + (",%.4f" % Precision[I]) + (",%.4f" % Recall[I]) + (",%.4f" % NDCG[I]) + (",%.4f" % MRR[I]) + "," + time + '\n'
        
        f.write(resultLine)
        f.close()

# get current time
def get_current_time():
    now = time.gmtime(time.time())
    
    if now.tm_mon < 10:
        month = "0"+str(now.tm_mon)
    else:
        month = str(now.tm_mon)
        
    if now.tm_mday < 10:
        day = "0"+str(now.tm_mday)
    else:
        day = str(now.tm_mday)
    
    if now.tm_hour < 10:
        hour = "0"+str(now.tm_hour)
    else:
        hour = str(now.tm_hour)
    
    if now.tm_min < 10:
        minute = "0"+str(now.tm_min)
    else:
        minute = str(now.tm_min)
        
    if now.tm_sec < 10:
        second = "0"+str(now.tm_sec)
    else:
        second = str(now.tm_sec)
        
    return month+day+"."+hour+minute+second

# get current time
def get_current_day():
    now = time.gmtime(time.time())
    
    if now.tm_mon < 10:
        month = "0"+str(now.tm_mon)
    else:
        month = str(now.tm_mon)
        
    if now.tm_mday < 10:
        day = "0"+str(now.tm_mday)
    else:
        day = str(now.tm_mday)
        
    return month+day

# Get batch data from training set
def get_batch_data(file, index, size):  # 1,5->1,2,3,4,5
    user = []
    item = []
    label = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip()
        line = line.split()
        user.append(int(line[0]))
        user.append(int(line[0]))
        item.append(int(line[1]))
        item.append(int(line[2]))
        label.append(1.)
        label.append(0.)
    return user, item, label

#Get user batch length 
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
        
def logFunction(be, d, m, SpecialParamStrs, COUNT, TOTAL, t1, t2, Server):
    logFile = be+"_"+d+"_"+m+"_process_log."+Server+".txt"
    f = open(logFile, 'a')
    now = time.localtime()
    nowtime = "[%02d:%02d:%02d]" % (now.tm_hour, now.tm_min, now.tm_sec)
    log = (nowtime+' ['+m+'] of '+SpecialParamStrs+', '+str(COUNT)+'/'+str(TOTAL)+' th process complete within '+str(t2-t1)+' seconds.'+'\n')
    f.write(log)
    f.close()
        