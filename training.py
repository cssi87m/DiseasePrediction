from preprocess import *
#  Import essential models  
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

""" Defining scoring metric for cross-validation
    The function takes 3 arguments:
    X: data that we use to predict the result 
    y: the ideal result
    estimator: method used for estimating"""

# # Initiate a scorer callable object 
# def cv_scoring(estimator, X, y):
#     return accuracy_score(y, estimator.predict(X))
# # Initializing model
# models={'SVC': SVC(),
#        'Gaussian_NB': GaussianNB(),
#        'Random_forest': RandomForestClassifier(random_state=18)}
# # Producing cross validation score for the models
# for model_name in models:
#     model = models[model_name]
#     model.fit(X, y)
#     scores =cross_val_score(model, X, y, cv = 10,
#                              n_jobs = -1,
#                              scoring = cv_scoring)

''' Buid a robust classifier by combining  model for training'''
# SVC
svc=SVC()
svc.fit(X_train, y_train)
# preds_svc=svc.predict(X_test)
# print(f"Accuracy on train data by SVM Classifier: {accuracy_score(y_train, svc.predict(X_train))*100}")
# print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, preds_svc)*100}")
# cf_matrix = confusion_matrix(y_test, preds_svc)
# plt.figure(figsize=(12,8))
# sns.heatmap(cf_matrix, annot=True)
# plt.title("Confusion Matrix for SVM Classifier on Test Data")
# plt.show()
# GaussianNB
gaussian=GaussianNB()
gaussian.fit(X_train, y_train)
# preds_gaussian=gaussian.predict(X_test)
# print(f"Accuracy on train data by GaussianNB Classifier:\
#        {accuracy_score(y_train, gaussian.predict(X_train))*100}")
# print(f"Accuracy on test data by GaussianNB Classifier:\
#       {accuracy_score(y_test, preds_gaussian)*100}")
# cf_matrix = confusion_matrix(y_test, preds_gaussian)
# plt.figure(figsize=(12, 8))
# sns.heatmap(cf_matrix, annot=True)
# plt.title("Confusion Matrix for GaussianNB Classifier on test data")
# plt.show()
# RandomForest
random_forest = RandomForestClassifier(random_state=24)
random_forest.fit(X_train, y_train)
# preds_random_forest= random_forest.predict(X_test)
# print(f"Accuracy on train data by RandomForest Classifier:\
#        {accuracy_score(y_train, random_forest.predict(X_train))*100}")
# print(f"Accuracy on test data by RandomForest Classifier:\
#       {accuracy_score(y_test, preds_random_forest)*100}")

