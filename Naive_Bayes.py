import csv
import random
import math
import cv2 as cv
import glob
import numpy as np
import math
from numpy import reshape
import matplotlib.pyplot as plt



def readImages():
    images = {}
    for file in sorted(glob.glob('Train/*.jpg')):
        if file[8] not in images:
            images[file[8]] = []
        images[file[8]].append(cv.imread(file, 0)/255)

    return images


def readTests():
    images = {}
    for file in sorted(glob.glob('Test/*.jpg')):
        if file[7] not in images:
            images[file[7]] = []
        images[file[7]].append(cv.imread(file, 0)/255)
    images1 = {}    
    for key,value in images.items():
        images1[key] = []
        for val in value:
            val = np.asarray(val)
            val = val.flatten()
            images1[key].append(val)

    return images1


def mean(images):
    mean = {}
    for key, value in images.items():
        mean[key] = np.zeros(value[0].shape)
        for val in value:
            mean[key] = np.add(mean[key], val)
        mean[key] = mean[key]/7
        mean[key] = mean[key].flatten()
    return mean


def stdev(images):
    std = {}
    for key, value in images.items():
        stdev = np.std(value, axis=0)
        stdev1 = stdev.flatten()
        std[key] = stdev1
    return std


def summarize(images):
    summary = {}
    mean1 = mean(images)
    stdev1 = stdev(images)
    for key, value in images.items():
        summary[key] = (mean1[key], stdev1[key])
    return summary


def calculateProbability(x, mean, stdev):
    if stdev == 0 :
        return 0.1
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        mean = classSummaries[0]
        stdev = classSummaries[1]
        for i in range(len(mean)):
            
            mean1 = mean[i]
            stdev1 = stdev[i]
            x = inputVector[i]
            prob = calculateProbability(x, mean1, stdev1)
            if prob <= 0.01 :
                prob = 0.1
            probabilities[classValue] *= prob

    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    pred = []
    for key, value in testSet.items():
        pred = []
        for i in range(len(value)):
            result = predict(summaries, value[i])
            pred.append(result)
        predictions.append(pred)
       
    return predictions

def plot(predictions,testSet):
    accuracy = np.zeros(26,)
    objects = []
    i = 0
    for key,value in testSet.items():
        objects.append(key)
        for p in predictions[i]:
            if p == key:
                accuracy[i]+=1
        i+=1        
    print(accuracy)            
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, accuracy, align='center', alpha=0.5)            
    plt.xticks(y_pos, objects)
    plt.title('Accuracy')
    plt.savefig('Accuracy.jpg') 
    plt.show()


def main():
    images = readImages()
    summa = summarize(images)
    test = readTests()
    gp = getPredictions(summa,test)
    plot(gp,test)
    print(gp)


main()
