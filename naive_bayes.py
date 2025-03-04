#-------------------------------------------------------------------------
# AUTHOR: Jairus Legion
# FILENAME: naive_bayes.py
# SPECIFICATION: Uses a Naive Bayes Classifier to predict whether tennis will be played
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
#Reading the training data in a csv file
#--> add your Python code here
db = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row)
#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
featureMap = {
    'Outlook': {'Sunny': 1, 'Overcast': 2, 'Rain': 3},
    'Temperature': {'Hot': 1, 'Mild': 2, 'Cool': 3},
    'Humidity': {'High': 1, 'Normal': 2},
    'Wind': {'Weak': 1, 'Strong': 2}
}

X = []
for row in db:
    features = []
    features.append(featureMap['Outlook'][row[1]])
    features.append(featureMap['Temperature'][row[2]])
    features.append(featureMap['Humidity'][row[3]])
    features.append(featureMap['Wind'][row[4]])
    X.append(features)
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in db:
    if row[-1] == "Yes":
        Y.append(1)
    else:
        Y.append(0)
#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
testData = []
with open('weather_test.csv', 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i > 0:
            testData.append(row)
#Printing the header os the solution
#--> add your Python code here
print("Day, Outlook, Temperature, Humidity, Wind, PlayTennis, Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in testData:
    testFeats = []
    testFeats.append(featureMap['Outlook'][row[1]])
    testFeats.append(featureMap['Temperature'][row[2]])
    testFeats.append(featureMap['Humidity'][row[3]])
    testFeats.append(featureMap['Wind'][row[4]])
    probabilities = clf.predict_proba([testFeats])[0]
    maxProb = max(probabilities)
    if maxProb >= 0.75:
        if probabilities[1] > probabilities[0]:
            prediction = 'Yes'
        else:
            prediction = 'No'
        print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {prediction}, {maxProb:.2f}")

