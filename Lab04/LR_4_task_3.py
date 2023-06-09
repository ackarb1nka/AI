import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'n_estimators': [25, 50, 100, 250], 'max_depth': [4]}
]
metrics = ['precision_weighted', 'recall_weighted']
for metric in metrics:
    print('Searching optimal parameters for', metric)
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)
    print("Grid scores for the parameter grid:")
"""""    for params, avg_score, _ in classifier.best_score_: classifier.best_score_: - Не працює 
        print(params, '-->', round(avg_score, 3))
        print('Best parameters:', classifier.best_params_)
"""""
y_pred = classifier.predict(X_test)
print('Performance report')
print(classification_report(y_test, y_pred))
