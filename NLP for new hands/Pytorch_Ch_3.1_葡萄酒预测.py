
# coding: utf-8

# In[54]:


import torch
import csv
import numpy as np
#初步读取数据
wine_path = 'C:\\Users\\dcy2015\\workspace\\winequality-white.csv'
wine_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter = ';', skiprows = 1) #忽略第一行，读取数据
col_list = next(csv.reader(open(wine_path),delimiter = ';')) #获取标签

#划分数据与标签
wineq = torch.from_numpy(wine_numpy)
data = wineq[:,:-1] #获取葡萄酒品质以外的数据
target = wineq[:,-1].long() #将分数化成整数标签

#进行独热编码
target_onehot = torch.zeros(target.shape[0],10)
target_onehot.scatter_(1,target.unsqueeze(1),1.0) #_方法原地修改张量
#target_onehot

#进行归一化
#data_mean = torch.mean(data,dim = 0)
#data_var = torch.var(data,dim = 0)
#data_normalized = (data - data_mean)/torch.sqrt(data_var)

#找出哪些行对应的葡萄酒口味不高于三分
#bad_indexes = torch.le(target,3) 
#bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()
#bad_data = data[bad_indexes]

bad_data = data[torch.le(target,3)]
mid_data = data[torch.gt(target,3)&torch.lt(target,7)]
good_data = data[torch.ge(target,7)]

bad_mean = torch.mean(bad_data,dim = 0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

#for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
#    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
    
total_sulfur_threshold = 141.83 #设定好酒二氧化硫阈值
total_sulfur_data = data[:,6] #取第六列二氧化硫值
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold) #基于二氧化硫阈值预测好酒索引

actual_indexes = torch.gt(target, 5) #以得分5为基准获取真正好酒的索引
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
n_matches, n_matches / n_predicted, n_matches / n_actual #准确率和召回率

