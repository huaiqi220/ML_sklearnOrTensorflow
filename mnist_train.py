import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier

mnist_data = fetch_openml("mnist_784")

# x,y= mnist_data["data"],mnist_data["target"]
# test = x[0]
# image = test.reshape(28 ,28)
# plt.imshow(image,cmap=matplotlib.cm.binary,
#            interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y[0])

x_train , x_test, y_train, y_test = mnist_data["data"][:60000],mnist_data["data"][60000:],mnist_data["target"][:60000],mnist_data["target"][60000:]

shuffle_index = np.random.permutation(60000)
# 打乱顺序
x_train, y_train = x_train[shuffle_index] , y_train[shuffle_index]

import numpy as np
noise1 = np.randint(0,100,(len(x_train),784))
noise2 = np.randint(0,100,(len(x_test),784))

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]




y_train_5 = (y_train == '5')
print(y_train_5)
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train,y_train_5)

print(sgd_clf.predict([x_train[0]]))

from sklearn.model_selection import cross_val_score
# cv3 means K-fold 折叠验证
score = cross_val_score(sgd_clf,x_train,y_train_5,cv=3,scoring="accuracy")
print(score)
'''
[0.85535 0.94995 0.95985]
分别为三次折叠验证的准确率
此处的非5数据集为偏斜数据集，即使直接猜不是5，也有百分之90正确率
所以此处准确率并不具有代表性
'''
# 评估分类器最好的方法是混淆矩阵
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,x_train,y_train_5,cv=3)
#
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_train_5,y_train_pred))

# 计算精度和召回率
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(recall_score(y_train_5,y_train_pred))

# 精度和召回率统一组成F1值
from sklearn.metrics import f1_score
print(f1_score(y_train_5,y_train_pred))

# 控制阈值，绘制精度/召回率曲线
y_scores = cross_val_predict(sgd_clf,x_train,y_train_5,cv=3,method="decision_function")
#
# from sklearn.metrics import precision_recall_curve
# precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
#
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label = "Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label = "precision")
    plt.xlabel("threshold")
    plt.legend(loc = "upper left")
    plt.ylim([0,1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


#
#
# plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
# plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# 用随机森林来拟合这个模型
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,x_train,y_train_5,cv=3,method="predict_proba")
y_scores_forest = y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="upper right")
plt.show()
