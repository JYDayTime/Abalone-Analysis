import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import nor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pycaret.classification import *

# This is a python file that can learn and predict probabilities from
# learning the data set.
# This code is very much based on the project file

#Set parameters
targetCSV  = 'Begin'#'50_50' 
normalizationState = True
normalizationType =  'min-max' #'zscore, minmax, robust, are all categories'
displayState = True
randSeed = 45
#First step, is to load the data
if targetCSV == 'Begin':
    abalone = pd.read_csv('data.csv')

#2.1 Preprocessing
def normalizeData(abalone, groundTruthCol, normalizationType='min-max'):

    features = abalone.drop(groundTruthCol, axis=1)
    groundTruth = abalone[groundTruthCol]

    # Apply normalization
    if normalizationType == 'z-score':
        scaler = StandardScaler()
        normalizedFeatures = scaler.fit_transform(features)
    elif normalizationType == 'min-max':
        scaler = MinMaxScaler()
        normalizedFeatures = scaler.fit_transform(features)
    else:
        raise ValueError("Invalid normalization type. Use 'z-score' or 'min-max'.")

    normalizedOutput = pd.DataFrame(normalizedFeatures, columns=features.columns)
    normalizedOutput[groundTruthCol] = groundTruth

    return normalizedOutput

if normalizationState== True:
    df = normalizeData(abalone, groundTruthCol='Sex', normalizationType='min-max')


def distributeClasses(abalone, groundTruthCol, randomState=None, plotDistribution=True):
    X = df.drop(groundTruthCol, axis=1)
    y = df[groundTruthCol]
    
    # Plot Class Distribution of the X and Y
    if plotDistribution:
        print("Class distribution:")
        print(y.value_counts())
        y.value_counts().plot(kind='bar', title='Class Distribution')
        plt.show()


    # Convert the result back to a DataFrame
    outPutClass = pd.DataFrame(X, columns=X.columns)
    outPutClass[groundTruthCol] = y
        
    return outPutClass

if targetCSV == 'Begin':
    df = distributeClasses(abalone,  groundTruthCol='Sex', randomState=None, plotDistribution=True)

#Setting up Machine Learning Process
#Splitting 70/15/15
trainValData, testData = train_test_split(abalone, testSize=0.15, randomState=randSeed)
trainData, valData = train_test_split(trainValData, testSize=0.25, randomState=randSeed)

#3.2 Model Creation
exp = ClassificationExperiment()
clf = exp.setup(trainData, target='Sex', sessionID=randSeed)
print(clf)

best = compare_models()
exp.compare_models()
# Create and tune a model on the training data
model = create_model('randModel')  # 'random model' is what we want to create from the algorithm
tunedModel = tune_model(model)

# Evaluate the model on the validation data
validPredictions = predict_model(tunedModel, data=valData)
print(validPredictions)

# Finalize the model on the combined train and validation sets
finalModel = finalize_model(tunedModel)

# Save the final model
save_model(finalModel, 'Abalone_Model')
# Evaluate the final model on the test data
finalTestPredictions = predict_model(finalModel, data=testData)
print(finalTestPredictions)
