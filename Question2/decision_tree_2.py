# -------------------------------------------------------------------------
# AUTHOR: Jairus Legion
# FILENAME: decision_tree_2.py
# SPECIFICATION: Trains a decision tree classifier using multiple csv files and tests accuracy over 10 iterations
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 Hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    featureMap = {}
    for col in range(len(dbTraining[0]) - 1):
        values = set()
        for row in dbTraining:
            values.add(row[col])
        sortedVals = sorted(values)
        temp = {}
        for i, val in enumerate(sortedVals):
            temp[val] = i
        featureMap[col] = temp

    for row in dbTraining:
        featureNums = []
        for col in range(len(dbTraining[0]) - 1):
            featureNums.append(featureMap[col][row[col]])

        X.append(featureNums)
    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    for row in dbTraining:
        if row[-1] == "Yes":
            Y.append(1)
        else:
            Y.append(2)
    # Loop your training and test tasks 10 times here
    totalAccuracy = 0
    for i in range(10):
        # Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTAest
        # --> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile2:
            reader2 = csv.reader(csvfile2)
            for j, row in enumerate(reader2):
                if j > 0:  # skipping the header
                    dbTest.append(row)
        correctTruths = 0
        totalPredictions = len(dbTest)
        for data in dbTest:
            testFeatures = []
            for col in range(len(data) - 1):
                testFeatures.append(featureMap[col][data[col]])

    # Transform the features of the test instances to numbers following the same strategy done during training,
    # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
    # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
    # --> add your Python code here
            prediction = clf.predict([testFeatures])[0]
    # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
    # --> add your Python code here
            if data[-1] == "Yes":
                trueLabel = 1
            else:
                trueLabel = 2

            if prediction == trueLabel:
                correctTruths += 1
# Find the average of this model during the 10 runs (training and test set)
# --> add your Python code here
        accuracy = correctTruths / totalPredictions
        totalAccuracy += accuracy
# Print the average accuracy of this model during the 10 runs (training and test set).
# Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
# --> add your Python code here
    avgAccuracy = totalAccuracy / 10
    print(f"Final accuracy when training on {ds}: {avgAccuracy}")
