from sklearn import svm
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time

class Error(Exception):
    """Timer error"""

class Timer:
    def __init__(self):
        self.startTime = None

    def start(self):
        """Start a new timer"""
        if self.startTime is not None:
            raise Error(f"Timer is running.")

        self.startTime = time.perf_counter()

    def stop(self,type):
        """Stop the timer, and report the time"""
        if self.startTime is None:
            raise Error(f"Timer is not running.")

        Time = time.perf_counter() - self.startTime
        self.startTime = None
        print(f"{type}: {Time:0.4f} seconds")
        
        

print("\nBenchmark for 'Shuttle' dataset which is used for learning purpose\n")
#Load Train and Test datasets
train=pd.read_table('shuttle.trn', sep=' ', header=None)
test=pd.read_table('shuttle.tst', sep=' ', header=None)
t=Timer()
# Data analysis
#train data
X = train.iloc[:, :-1]
X.head()
train_y=train[9]
train_x=train.drop([9],axis=1)

#test data
y= test.iloc[:, :-1]
y.head()
test_y=test[9]
test_x=test.drop([9],axis=1)
s=input("Display statistics of 'Shuttle' dataset yes or no:")
if s=="yes":
    test_y.hist()# most of features are related to  1, 4 and 5 classes according to the histogram
    majorsamples = test[9].isin([1,4,5])
    print("\nThe number of samples in of 1,4 and 5 classes for the test smaple is ",len(test[majorsamples]) / len(test),"%")
    minorsamples = test[9].isin([2,3,6,7])
    print("\nThe number of samples in of 2,3,6 and 7 classes for the test smaple is ",len(test[minorsamples]) / len(test),"%")
    # therefore the classes 1,4 and 5 account for 99.6% of all samples available,
    # we can consider classes 2,3,6 and 7 as outsmaples.

ML=input("Please select machine learning algorithm form (L.Reg,KNN ,Decision Tree and SVM):")
#logistic Regression
if ML=="L.Reg":
    rand=int(input("Please enter the value of random_state :")) 
    t.start()
    logistic=LogisticRegression(C=10**-5,random_state=rand) #after trying 2 values of C
    logistic.fit(train_x, train_y)
    t.stop("\nTraining time")
    t.start()
    predicted=logistic.predict(test_x)
    result=accuracy_score(test_y,predicted)
    t.stop("Testing time")
    print("Accuracy of the Logistic Regression is ", result)
   

elif ML=="KNN":
#KNN 
#Calculating Euclidean Distance
# calculate the Euclidean distance between two vectors
#euclidean_distances(train_x,train_x)
    nn=int(input("Please enter the value of number of neighbors:\n"))
    t.start()
    neighbors = KNeighborsClassifier(n_neighbors=nn)
    neighbors.fit(train_x, train_y)
    t.stop("\nTraining time")
    t.start()
    predicted=neighbors.predict(test_x)
    result=accuracy_score(test_y,predicted)
    t.stop("Testing time")
    print("Accuracy of the KNN  is ", result)
    

# Linear and non linear SVM 
elif ML=="SVM":
    sv=input("Please enter select between(Linear and non linear) Kernel:\n")
    if sv=="Linear":
        rand=int(input("Please enter the value of random_state :\n"))
        t.start()
        svmLinear = svm.LinearSVC(C=10**-5,dual=True, random_state=rand) #so far random_state=0 and dual=true
        svmLinear.fit(train_x, train_y)
        t.stop("\nTraining time")
        t.start()
        predicted= svmLinear.predict(test_x)
        result=accuracy_score(test_y,predicted)
        t.stop("Testing time")
        print("Accuracy of the linear SVM is ", result)
    elif sv=="non linear":
         g=float(input("Please enter the value of gamma :\n"))
         t.start()
         c= SVC(gamma=g, C=100.)
         c.fit(train_x, train_y)
         t.stop("\nTraining time")
         t.start()
         predicted= c.predict(test_x)
         result=accuracy_score(test_y,predicted)
         t.stop("Testing time")
         print("Accuracy of the non linear SVM is ", result)
#Decision Tree        
elif ML=="Decision Tree":  
    rand=int(input("Please enter the value of random_state :\n"))
    t.start()     
    Dec= DecisionTreeClassifier(random_state=rand)
    Dec.fit(train_x, train_y)
    t.stop("\nTraining time")
    t.start()
    predicted= c.predict(test_x)
    result=accuracy_score(test_y,predicted)
    t.stop("Testing time")
    print("Accuracy of the Decision Tree SVM is ", result)


