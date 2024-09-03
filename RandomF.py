#程序说明：本程序实际上不是以脚本方式运行，而是在python交互式界面运行，该文件只作为保存用，也可以使用Jupter Notebook打开
#主要目的是方便数据保存，由于数据过大，一次读取需时过长，为节约时间，采用这种方式
#本程序是随机森林算法的实现，利用了sklearn的RandomForestClassifier接口实现，从前到后依次包括
#库导入部分，部分预处理，默认参数模型训练，输出结果以及绘图，搜索最佳参数模型训练，输出最优结果以及绘图
#详细解释请阅读对应行注释



#导入sklearn接口
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt

#程序主体在console运行，这样每次读取数据不丢失，省去读取数据时间，直接打开终端python交互界面，复制代码粘贴到终端上即可。
#有部分代码在终端运行如下(数据读取完成以后),主要功能是对数据预处理，包括复制正例，补充缺失值等
'''
>>>dt = dt.fillna(0)
>>>cf = dt[dt['yy']==1]
>>>cf = cf.append(cf)
>>>cf = cf.append(cf)
>>>cf = cf.append(cf)
>>>cf = cf.append(cf)   #正例翻16倍
>>>dt = dt.append(cf)
'''



#使用pandas库接口读取数据
dt = pd.read_excel(r'E:\programing\Globin_Python\数据挖掘\data_an.xlsx')
print(dt)
print('读取完成')

#切分属性，x为预测用属性，即drop掉不需要的属性，label为监督值
x = dt.drop(['yy'], axis=1) 
label = dt['yy'] 

#利用sklearn接口切分数据
xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=0.3)
xtrain.head()

#以默认参数进行训练
rfc = RandomForestClassifier()   
rfc = rfc.fit(xtrain, ytrain)
score_r = rfc.score(xtest, ytest)
print(score_r)

# 随机森林 预测测试集
y_test_proba_rfc = rfc.predict_proba(xtest)
false_positive_rate_rfc, recall_rfc, thresholds_rfc = roc_curve(ytest, y_test_proba_rfc[:, 1])  
# 随机森林 AUC指标
roc_auc_rfc = auc(false_positive_rate_rfc, recall_rfc)     
#输出AUC曲线
plt.plot(false_positive_rate_rfc, recall_rfc, color='orange', label='AUC_rfc=%0.3f' % roc_auc_rfc)  
plt.legend(loc='best', fontsize=15, frameon=False)  
plt.plot([0, 1], [0, 1], 'r--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.0])  
plt.ylabel('Recall')  
plt.xlabel('Fall-out')  
plt.show()

#特征重要度输出
feature_name = x.columns
rfc.feature_importances_
a = [*zip(feature_name,rfc.feature_importances_)]

def takeSecond(elem):
    return elem[1] #以第二个元素（特征重要度）为排序依据
a.sort(key=takeSecond,reverse=True)  #排序
x = []             #列表拆分为index和数据部分
y = []
for i,j in a:
    x.append(i)
    y.append(j)
plt.bar(x,y)     #柱状图
plt.show()


# 定义空列表，用来存放每一个基学习器数量所对应的AUC值
superpa = []
# 循环200次
for i in [50,100,150,200,250,300,400,500,600,700,900,1200,1400,1700,2000,5000,7500,10000,50000]:
    rfc = ensemble.RandomForestClassifier(n_estimators=i+1, class_weight='balanced',random_state=50, n_jobs=10)
    rfc = rfc.fit(xtrain, ytrain)   # 拟合模型
    
    y_test_proba_rfc = rfc.predict_proba(xtest)   # 预测测试集
    false_positive_rate_rfc, recall_rfc, thresholds_rfc = roc_curve(ytest, y_test_proba_rfc[:, 1])  
    roc_auc_rfc = auc(false_positive_rate_rfc, recall_rfc)   # 计算模型AUC
    
    print(i,roc_auc_rfc)
    superpa.append(roc_auc_rfc)   # 记录每一轮的AUC值

print(max(superpa),superpa.index(max(superpa)))   # 输出最大的AUC值和其对应的轮数
rfc = RandomForestClassifier(n_estimators=superpa.index(max(superpa)), class_weight='balanced',random_state=50, n_jobs=10)   #重新获得对应轮数的模型
rfc = rfc.fit(xtrain, ytrain)
score_r = rfc.score(xtest, ytest)         #模型预测得分
print(score_r)
plt.figure(figsize=[20,5])              #绘图，整个过程模型良好程度变化趋势
plt.plot(range(1,20),superpa)
plt.show()

#plt设置部分，显示中文字体
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容

#特征重要性输出
feature_name = x.columns
rfc.feature_importances_
a =  [*zip(feature_name,rfc.feature_importances_)] #先整合，方便排序


def takeSecond(elem):
    return elem[1] #以第二个元素（特征重要度）为排序依据
a.sort(key=takeSecond,reverse=True)  #排序
x = []             #列表拆分为index和数据部分
y = []
for i,j in a:
    x.append(i)
    y.append(j)
plt.bar(x,y)     #柱状图
plt.show()