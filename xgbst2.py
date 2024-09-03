#程序说明：本程序实际上不是以脚本方式运行，而是在python交互式界面运行，该文件只作为保存用，也可以使用Jupter Notebook打开
#主要目的是方便数据保存，由于数据过大，一次读取需时过长，为节约时间，采用这种方式
#本程序是XGBoost算法实现，主要采用了xgboost库来实现，以及sklearn的数据切分接口，程序本身包括
#库导入，数据读取预处理，调参，模型训练，特征重要度输出以及绘图。
#详细解释请阅读对应行注释


#库导入部分
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV   #网格搜索调参接口

# 用pandas读入数据
data = pd.read_excel(r'E:\programing\Globin_Python\数据挖掘\data_an.xlsx')
print(data)

#部分数据预处理同RandomF.py文件，直接复制即可，这里不展示，注意变量名使用data而非dt

# 做数据切分，切分测试集以及训练集
train, test = train_test_split(data)

# 将原始数据转换成Dmatrix格式供模型读取
feature_columns = ['性别','年龄','客户等级','客户风险等级','理财购买金额','理财占比','月日均理财持有','近3月存入','近三月汇出金额','近三月跨行转账次数','近3个月存款净增加额','卡近3月转账金额','八月末理财','八月末存款','八月末资产','七月末存款','七月末资产']
target_column = 'yy'
xgtrain = xgb.DMatrix(train[feature_columns].values, train[target_column].values)
xgtest = xgb.DMatrix(test[feature_columns].values, test[target_column].values)


#可选调参部分，直接输入到xgb.XGBClassfier(param_dist)即可
param_dist = {
'n_estimators':50,
'max_depth':50,
'learning_rate':0.1,
'subsample':0.7,
}


#模型训练,先使用默认参数尝试
xgb_model = xgb.XGBClassifier().fit(train[feature_columns],train[target_column].values)
# 使用模型预测测试集结果
preds = xgb_model.predict(xgtest)
# 判断准确率，看交叉验证结果
labels = xgtest.get_label()
print ('错误类为%f' % \
        (sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

#特征重要度
print('特征排序：')
feature_names=['性别','年龄','客户等级','客户风险等级','理财购买金额','理财占比','月日均理财持有','近3月存入','近三月汇出金额','近三月跨行转账次数','近3个月存款净增加额','卡近3月转账金额','八月末理财','八月末存款','八月末资产','七月末存款','七月末资产']
feature_importances = xgb_model.feature_importances_
indices = np.argsort(feature_importances)[::-1] 
for index in indices:
    print("特征 %s 重要度为 %f" %(feature_names[index], feature_importances[index]))

#特征重要度绘图
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.title("feature importances")
plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], color='b')
