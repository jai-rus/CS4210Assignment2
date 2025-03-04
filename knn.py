#-------------------------------------------------------------------------
# AUTHOR: Jairus Legion
# FILENAME: knn.py
# SPECIFICATION: Utilizes k-nearest neighbors to classify email instances and calculates the error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

errorCount = 0
#Loop your data to allow each instance to be your test set
for i, testInstance in enumerate(db):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for j, instance in enumerate(db):
        if i != j:
            features = []
            for value in instance[:-1]:
                features.append(float(value))
            X.append(features)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
            if instance[-1] == "ham":
                label = 0
            else:
                label = 1
            Y.append(label)
    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = []
    for value in testInstance[:-1]:
        testSample.append(float(value))
    if testInstance[-1] == "ham":
        trainValue = 0
    else:
        trainValue = 1
    truthVal = trainValue
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    classPredicted = clf.predict([testSample])[0]
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if classPredicted == truthVal:
        errorCount += 1
#Print the error rate
#--> add your Python code here
print(f"Error Rate: {1 - errorCount/len(db)}")





