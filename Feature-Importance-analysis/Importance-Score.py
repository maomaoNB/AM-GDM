## 输出高清图像
# config InlineBackend.figure_format = 'retina'
# matplotlib inline
## 图像显示中文的问题
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False#用来正常显示负号

import seaborn as sns 
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)#绘图函数，设置字体、风格和字体大小

import pandas as pd
pd.set_option("max_colwidth", 200)#设置最大列数，显示前200列

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import missingno as msno


from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.metrics import *
from io import StringIO
from sklearn.model_selection import GridSearchCV


import graphviz
import pydotplus
from IPython.display import Image  

## 忽略提醒
import warnings
y_train=pd.read_excel("D:/Code/横向论文代码（孕期糖尿病预测模型）/特征重要性分析/y_train.xlsx")
X_train=pd.read_excel("D:/Code/横向论文代码（孕期糖尿病预测模型）/特征重要性分析/X_train.xlsx")
X_val=pd.read_excel("D:/Code/横向论文代码（孕期糖尿病预测模型）/特征重要性分析/X_val.xlsx")
y_val=pd.read_excel("D:/Code/横向论文代码（孕期糖尿病预测模型）/特征重要性分析/y_val.xlsx")
train_x=X_train.columns
rfc1 = RandomForestClassifier(n_estimators = 100, # 树的数量
                              max_depth= 5,       # 子树最大深度
                              oob_score=True, 
                              class_weight = "balanced",
                              random_state=1)
rfc1.fit(X_train,y_train)
## 输出其在训练数据和验证数据集上的预测精度
rfc1_lab = rfc1.predict(X_train)
rfc1_pre = rfc1.predict(X_val)
importances = pd.DataFrame({"feature":train_x,
                            "importance":rfc1.feature_importances_})
importances = importances.sort_values("importance",ascending = True)
importances.plot(kind="barh",figsize=(10,6),x = "feature",y = "importance",
                 legend = False)
plt.xlabel("Iportance Score")
plt.ylabel("")
plt.title("Random Forest Classifler")
plt.grid()
plt.show()
