"""
Massimo Zimmerman
Dataset Classification Project
PRE-PLAN
"""

# Importing/Activating Libraries
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

# Importing Dataset into Program
"""
PLAN 1
"""

"""
The CSV/Dataset is indexed/labeled as follows:
'ID', 'Diagnosis (B/M)'
'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension'
'StDev Radius', 'StDev Texture', 'StDev Perimeter', 'StDev Area', 'StDev Smoothness', 'StDev Compactness', 'StDev Concavity', 'StDev Concave Points', 'StDev Symmetry', 'StDev Fractal Dimension',
'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
"""
# Processing the Dataset into a Pandas Dataframe
raw_data = pd.read_csv(r'D:/Downloads - HDD//wdbc.dataset', header=None)
data = raw_data.iloc[:,2:]
data_num = data.shape[0]

# Converting the Diagnosis Column from (B/M) to (0/1) 
# This allows for the entire Dataframe to be of type 'float' or 'int'
for i in range(0,data_num):
    label = raw_data.iloc[i,1]
    if label == 'B':
        raw_data.iloc[i,1] = 0
    else:
        raw_data.iloc[i,1] = 1

raw_data.iloc[:,1] = raw_data.iloc[:,1].astype(float)


"""
PLAN 2
"""
# Partitioning Dataset into Training and Testing
target = raw_data.iloc[:,1]
train_ratio = 0.8
train_num = int(data_num * train_ratio)
train_data = data.iloc[:train_num,:]
test_data = data.iloc[train_num:,:]
train_target = target.iloc[:train_num]
test_target = target.iloc[train_num:]


"""
PLAN 3
"""
# Initialize the Results Matrix
results = []

# Setup Train Data and Test Data using SGDClassifier
sgd_model = linear_model.SGDClassifier()
sgd_model.fit(train_data, train_target)

# Setup Test Data and Test Data using SGDClassifier
test_model = sgd_model.predict(test_data)
r1 = sklearn.metrics.mean_squared_error(test_target, test_model)
results.append(r1)


"""
Plan 4
"""
# Printing Results from SGDClassifier Run
print ('SGD Classifier, MSE : %f' % r1)

# Printing Accuracy Matrix
accuracy = sklearn.metrics.accuracy_score(test_target, test_model)
print('Accuracy Score')
print (accuracy)
print('')

# Printing Precision Matrix
precision = sklearn.metrics.precision_score(test_target, test_model)
print('Precision Score')
print (precision)
print('')

# Printing Recall Matrix
recall = sklearn.metrics.recall_score(test_target, test_model)
print('Recall Score')
print (recall)
print('')

# Printing Confusion Matrix
confusion = sklearn.metrics.confusion_matrix(test_target, test_model)
print('Confusion Matrix')
print (confusion)
print('')


"""
PLAN 5
"""
# Generate Cross-Validated Estimates for Training Data
train_pred = cross_val_predict(sgd_model, train_data, train_target, cv=3, method="decision_function")
test_pred = cross_val_predict(sgd_model, test_data, test_target, cv=3, method="decision_function")
result_score = cross_val_score(sgd_model, train_data, train_target, cv=3, scoring="accuracy")
print('Cross Validation Result')
print(result_score)
print('')


"""
PLAN 6
"""
fpr_train, tpr_train, thresholds = sklearn.metrics.roc_curve(train_target, train_pred, pos_label = 1)
fpr_test, tpr_test, thresholds = sklearn.metrics.roc_curve(test_target, test_pred, pos_label = 1)

plt.figure()
plt.title('ROC Plot (Training Set)')
plt.plot(fpr_train, tpr_train)
plt.plot([0,1], [0,1], 'k--')
plt.ylabel('Hit Rate')
plt.xlabel('False Alarms')
plt.show()

plt.figure()
plt.title('ROC Plot (Testing Set)')
plt.plot(fpr_test, tpr_test)
plt.plot([0,1], [0,1], 'k--')
plt.ylabel('Hit Rate')
plt.xlabel('False Alarms')
plt.show()



