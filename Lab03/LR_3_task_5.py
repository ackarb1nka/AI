import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
#indices = np.argsort(X, axis=0)
#X = X[indices].reshape(-1, 1)
#y = y[indices].reshape(-1, 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= 0.5, random_state = 0)
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)

print(f"Mean absolute error = {round(mean_absolute_error(ytest, ypred), 2)}")
print(f"Mean squared error = {round(mean_squared_error(ytest, ypred), 2)}")
print(f"Regression coefficient = {round(regr.coef_[0][0], 2)}")
print(f"Regression intercept = {round(regr.intercept_[0], 2)}")
print(f"R2 score = {round(r2_score(ytest, ypred), 2)}")

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()

"""poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
ypred = lin_reg.predict(X_poly)

print(f"Mean absolute error = {round(mean_absolute_error(y, ypred), 2)}")
print(f"Mean squared error = {round(mean_squared_error(y, ypred), 2)}")
print(f"Regression coefficient = {round(regr.coef_[0][0], 2)}")
print(f"Regression intercept = {round(regr.intercept_[0], 2)}")
print(f"R2 score = {round(r2_score(y, ypred), 2)}")

fig, ax = plt.subplots()
ax.scatter(y, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()

"""