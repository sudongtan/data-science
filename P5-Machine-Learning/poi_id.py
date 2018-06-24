#!/usr/bin/python

import sys
import pickle
#!touch tools/__init__.py
#sys.path.insert(0, 'tools')

from tools.feature_format import featureFormat, targetFeatureSplit
from tools.tester import dump_classifier_and_data
from tools.tester import test_classifier

import pprint
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features_list = ['salary', 'deferral_payments', 
                           'total_payments', 
                           'bonus', 
                           'exercised_stock_options', 
                           'restricted_stock', 
                           'restricted_stock_deferred', 
                           'total_stock_value', 'expenses', 
                           'loan_advances', 
                           'other', 
                           'director_fees', 
                           'deferred_income', 
                           'long_term_incentive' ]

email_features_list = ['to_messages', 
                       'from_messages', 
                       'from_this_person_to_poi', 
                       'from_poi_to_this_person', 
                       'shared_receipt_with_poi']

features_list = ['poi'] + financial_features_list + email_features_list

### Load the dictionary containing the dataset
with open("data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict["TOTAL"]
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['LOCKHART EUGENE E']

### Task 3: Create new feature(s)

for person, features in data_dict.items():
    a = features["from_poi_to_this_person"]
    b = features["from_this_person_to_poi"]
    c = features['to_messages']
    d = features['from_messages']  
    
    if a == "NaN" or c == "NaN" or b == "NaN" or d == "NaN" or (c == 0 and d == 0): 
        features['total_poi_percentage'] = "NaN"
    else:
        features['total_poi_percentage'] = round(float(a+b)/(c+d),4) 

features_list_expanded = features_list + ['total_poi_percentage']

### Store to my_dataset for easy export below.
my_dataset = data_dict
#print len(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_expanded, sort_keys = True)
target, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Tuning parameters

# def tune_parameters(clf, parameters, folds=1000):
#     grid_search = GridSearchCV(clf, parameters, 
#                       cv=StratifiedShuffleSplit(target, folds, random_state=42),
#                       n_jobs=-1, scoring='f1', verbose = 1)
#     grid_search.fit(features, target)
    
#     best_model = grid_search.best_estimator_
    
#     print type(clf.steps[-1][1]).__name__
#     print grid_search.best_params_
#     print grid_search.best_score_
#     print test_classifier(best_model, my_dataset, features_list_expanded)


# tuning logistic regression    
# lr_clf_scaled = Pipeline(steps=[('kbf', SelectKBest()),
#                                 ('scaler', StandardScaler()),
#                                 ('clf', LogisticRegression())])

# parameters = {'kbf__k':range(1,21),
#               'clf__penalty': ['l1', 'l2'], 
#               'clf__C': np.logspace(-10, 0, 11), 
#               'clf__random_state': [0]} 

# tune_parameters(lr_clf_scaled, parameters)


#tuning SVM
# s_clf_scaled = Pipeline(steps=[('kbf', SelectKBest()),
#                                ('scaler', StandardScaler()), 
#                                ('clf', SVC())])

# parameters = {'kbf__k':range(1,21),
#               'clf__kernel':['rbf'], 
#               'clf__C':[0.1,1,10,100,1000], 
#               'clf__gamma': np.logspace(-5,0,6),
#               'clf__tol': [1e-1, 1e-2, 1e-4, 1e-5], 
#               'clf__class_weight': ['balanced']}  
    
# tune_parameters(s_clf_scaled, parameters)


# tuning decision tree
# dt_clf_pl = Pipeline(steps=[('kbf', SelectKBest()),
#                             ('clf', DecisionTreeClassifier())])

# parameters = {'kbf__k':range(1,21),
#               "clf__min_samples_leaf": [2, 6, 10, 12],
#               "clf__min_samples_split": [2, 6, 10, 12],
#               "clf__criterion": ["entropy", "gini"],
#               "clf__max_depth": [None, 5],
#               "clf__random_state": [0]}

# tune_parameters(dt_clf_pl, parameters)


#tuning random forest
# rf_clf_pl = Pipeline(steps=[('kbf', SelectKBest()),
#                             ('clf', RandomForestClassifier())])

# parameters = {'kbf__k':range(1,21),
#               'clf__max_depth': [None, 5, 10],
#               'clf__n_estimators': [10, 15, 20, 25],
#               'clf__random_state': [0]}

# tune_parameters(rf_clf_pl, parameters)



### Tuned algorithms
#Logistic Regression
lr_clf_tuned = Pipeline(steps=[('kbf', SelectKBest(k=10)),
                               ('scaler', StandardScaler()), 
                               ('clf', LogisticRegression(penalty = 'l2', 
                                                          C = 0.0001, 
                                                          random_state = 0))])

#SVM 
s_clf_tuned = Pipeline(steps=[('kbf', SelectKBest(k=20)),
                              ('scaler', StandardScaler()), 
                              ('clf', SVC(gamma= 0.001,
                                          C=1000,
                                          tol=0.0001,
                                          class_weight='balanced',
                                          kernel='rbf'))])                                                                         

#Naive Bayes (default parameters)
nb_clf_pl = Pipeline(steps=[('kbf', SelectKBest()),
                            ('clf', GaussianNB())])


#Decision Tree
dt_clf_tuned = Pipeline(steps=[('kbf', SelectKBest(k = 15)),
                               ('clf', DecisionTreeClassifier(criterion='entropy', 
                                                               max_depth=None, 
                                                               min_samples_leaf=12,
                                                               min_samples_split=2, 
                                                               random_state=0))])

#Random Forest
rf_clf_tuned = Pipeline(steps=[('kbf', SelectKBest(k = 3)),
                               ('clf', RandomForestClassifier(max_depth=None, 
                                                               random_state=0, 
                                                               n_estimators=25))])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!


lr_clf_final = Pipeline(steps=[('scaler', StandardScaler()), 
                               ('clf', LogisticRegression(penalty = 'l2', 
                                                          C = 0.0001, 
                                                          random_state = 0))])

best_features = ['exercised_stock_options', 
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'deferred_income', 
                 'long_term_incentive',
                 'restricted_stock',
                 'total_payments',
                 'shared_receipt_with_poi',
                 'loan_advances']

selected_features_list = ['poi'] + best_features

def evaluation(clf, features_list, folds = 1000):
    """calculate the precision, recall and f1 of a classifier, using k-fold"""
    
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    target, features = targetFeatureSplit(data)

    precision = []
    recall = []
    f1 = []
    
    cv = StratifiedShuffleSplit(target, folds, random_state = 42)
    
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( target[ii] )
        for jj in test_idx:
            features_test.append(features[jj] )
            labels_test.append( target[jj] )
    
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        f1.append(f1_score(labels_test, predictions))
    
    print "Precision: ", round(np.mean(precision),4)
    print "Recall: ", round(np.mean(recall),4)
    print "f1: ", round(np.mean(f1),4)

evaluation(lr_clf_final, selected_features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(lr_clf_final, my_dataset, selected_features_list)