from data_deal import get_data
from data_deal import deal_data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

h_tr,h_label = deal_data.get_final_data()

# lin_reg = LinearRegression()
# lin_reg.fit(h_tr,h_label)
some_data = h_tr.iloc[:5]
some_labels = h_label[:5]
# print("Predictions:\t",lin_reg.predict(some_data))
# print("labels:\t\t",list(some_labels))
# housing_pred = lin_reg.predict(some_data)
# lin_mse = mean_absolute_error(some_labels,housing_pred)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
'''
Predictions:	 [207533.05517344 322908.0066498  205636.93169928  75429.96006053
 188528.17569525]
labels:		 [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
'''

tree_reg = DecisionTreeRegressor()
tree_reg.fit(h_tr,h_label)
housing_predt = tree_reg.predict(some_data)
tree_mse = mean_absolute_error(some_labels,housing_predt)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)





