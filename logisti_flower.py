from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
iris = datasets.load_iris()

x = iris["data"][:,3:]
y = (iris["target"] == 2).astype(np.int)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x,y)

x_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(x_new)

plt.plot(x_new,y_proba[:,1],"g--",label="Iris-Virginica")
plt.plot(x_new,y_proba[:,0],"b--",label="Not Iris")

plt.show()



