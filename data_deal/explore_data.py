from data_deal import get_data
from matplotlib import pyplot as plt
from data_deal import deal_data

#
# housing = get_data.load_housing_data()
# housing.hist(bins=50,figsize=(20,15))
# plt.show()


def plt_scatter_data():
    s_tr,s_te = deal_data.deal_data_by_sss()
    housing = s_tr.copy()
    housing.plot(kind= "scatter",x = "longitude", y = "latitude", alpha = 0.4,
                 s = housing["population"] / 100 , label = "population",
                 c = "median_house_value",cmap = plt.get_cmap("jet"),colorbar = True,)

    plt.legend()
    plt.show()



plt_scatter_data()