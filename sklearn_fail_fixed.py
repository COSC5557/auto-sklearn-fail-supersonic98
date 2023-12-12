import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)
X = np.array(X)
y = np.array(y)
# enc = OneHotEncoder(handle_unknown='ignore')
# X = enc.fit_transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat_clf = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat_clf))


from autosklearn.classification import AutoSklearnClassifier
automl_no_cv = AutoSklearnClassifier(time_left_for_this_task=300)
automl_cv = AutoSklearnClassifier(time_left_for_this_task=300,
            resampling_strategy="cv", resampling_strategy_arguments={"train_size": 0.80,  "shuffle": True,  "folds": 5})


automl_no_cv.fit(X_train, y_train)
y_hat_no_cv = automl_no_cv.predict(X_test)
print("AutoML Accuracy without CV", sklearn.metrics.accuracy_score(y_test, y_hat_no_cv))
print(automl_no_cv.sprint_statistics())


automl_cv.fit(X_train, y_train)
y_hat_cv = automl_cv.predict(X_test)
print("AutoML Accuracy with CV", sklearn.metrics.accuracy_score(y_test, y_hat_cv))
print(automl_cv.sprint_statistics())


""" Prints
RF Accuracy 0.67
AutoML Accuracy without CV 0.6375
auto-sklearn results:
  Dataset name: 56fec466-9878-11ee-9b38-a4bb6dbe60de
  Metric: accuracy
  Best validation score: 0.691919
  Number of target algorithm runs: 86
  Number of successful target algorithm runs: 81
  Number of crashed target algorithm runs: 5
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

AutoML Accuracy with CV 0.6775
auto-sklearn results:
  Dataset name: 088c6c11-9879-11ee-9b38-a4bb6dbe60de
  Metric: accuracy
  Best validation score: 0.686405
  Number of target algorithm runs: 24
  Number of successful target algorithm runs: 20
  Number of crashed target algorithm runs: 0
  Number of target algorithms that exceeded the time limit: 4
  Number of target algorithms that exceeded the memory limit:
"""
