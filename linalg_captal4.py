import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)
x_b = np.c_[np.ones((100, 1)), x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta_best)

# iPython代码
# Python 3.7.9 | packaged by conda-forge | (default, Dec  9 2020, 20:36:16) [MSC v.1916 64 bit (AMD64)] on win32
# runfile('D:/开发工具及代码demo/first_ml/linalg_captal4.py', wdir='D:/开发工具及代码demo/first_ml')
# [[4.40694524]
#  [3.0620998 ]]
# x_new = np.array([[0],[2]])
# x_new_b = np.c_[np.ones((2,1)),x_new]
# y_predict = x_new_b.dot(theta_best)
# y_predict
# Out[6]:
# array([[ 4.40694524],
#        [10.53114484]])


# plt.plot(x_new,y_predict,"r-")
# Out[8]: [<matplotlib.lines.Line2D at 0x23d3a4edec8>]
# plt.plot(x,y,"b.")
# Out[9]: [<matplotlib.lines.Line2D at 0x23d3a51d508>]
# plt.axis([0,2,0,15])
# Out[10]: (0.0, 2.0, 0.0, 15.0)
# plt.show()

eta = 0.1  # learning rate
n_iterations = 1000
m = 100
theta = np.random.rand(2, 1)
print(theta)

for iteration in range(n_iterations):
    gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradients

n_epochs = 5
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t + t1)


for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)


def plot_learning_curves(model, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


#
# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, x, y)


from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline((
    ("poly_features", PolynomialFeatures(degree= 10, include_bias=False)),
    ("sgd_reg", LinearRegression()),
))

plot_learning_curves(polynomial_regression, x ,y)
plt.show()
