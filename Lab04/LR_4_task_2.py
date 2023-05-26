import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
plt.figure()
scatter_params = {'s': 75, 'edgecolors': 'black', 'linewidths': 1}
plt.scatter(class_0[:, 0], class_0[:, 1], s= 75, facecolors= 'black', edgecolors= 'black', linewidths= 1, marker= 'x')
plt.scatter(class_1[:, 0], class_1[:, 1], s= 75, facecolors= 'white', edgecolors= 'black', linewidths= 1, marker= 'o')
plt.title('Вхідні дані')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if len(sys.argv) > 1:
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight': 'balanced'}
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test data')

class_names = ['Class-0', 'Class-1']
print('\n' + '#' * 40)
print('Classifier performance on training dataset')
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print('\n' + '#' * 40)
print('Classifier performance on test dataset')
print(classification_report(y_test, y_test_pred, target_names=class_names))
plt.show()
