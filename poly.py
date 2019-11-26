import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#In general within the work force, the more work experience an employee has,
#the greater the salary they would receive over time. In this model, I will be
#exploring the relationship between an employees salary and their years worked

#Training set
x_train = [[1], [3], [4], [7], [9]] #years worked
y_train = [[45000], [60000], [80000], [200000], [500000]] #Salary

#Testing set
x_test = [[1], [2], [5], [10]]
y_test = [[50000], [60000], [110000], [1000000]]

#Train the Linear Regression model and plot prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 10, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

#Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree = 2)

#Transform input data matrix into a new data matrix of degree 2
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

#Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

#Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='b', linestyle='--')
plt.title('Salary regressed on years worked')
plt.xlabel('Years worked')
plt.ylabel('Salary in dollars')
plt.axis([0, 10, 0, 1000000])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
