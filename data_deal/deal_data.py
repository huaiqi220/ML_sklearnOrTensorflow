from data_deal import get_data
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def test_set_check(identifier,test_ratio,hash):
    return hash(np.init64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data,test_ratio,id_column,hash = hashlib.md5()):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]

'''
上面这个函数没看懂
'''
# 纯随机抽样 --- 出现了巨大偏差
def get_train_test_set():
    housing = get_data.load_housing_data()
    housing_with_id = housing.reset_index()
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    return train_set,test_set

# 分层抽样 --- 最准确
def deal_data_by_sss():
    housing = get_data.load_housing_data()
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5 , 5.0 , inplace=True)
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_set , test_index in split.split(housing,housing["income_cat"]):
        strat_train_set = housing.loc[train_set]
        strat_test_set = housing.loc[test_index]
    for set in (strat_test_set,strat_train_set):
        set.drop(["income_cat"],axis = 1, inplace = True)
    return strat_train_set,strat_test_set


#求标准相关系数
def corr_data():
    s_tr, s_te = deal_data_by_sss()
    housing = s_tr.copy()
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))



# corr_data()
'''
median_house_value    1.000000
median_income         0.687160
total_rooms           0.135097
housing_median_age    0.114110
households            0.064506
total_bedrooms        0.047689
population           -0.026920
longitude            -0.047432
latitude             -0.142724
Name: median_house_value, dtype: float64
'''


def get_final_data():
    s_tr, s_te = deal_data_by_sss()
    housing = s_tr.drop("median_house_value", axis=1)
    housing_labels = s_tr["median_house_value"].copy()
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity",axis =1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X,columns = housing_num.columns)
    return housing_tr,housing_labels



