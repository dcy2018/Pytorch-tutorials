# coding:utf-8
#Filename:note.py
import types
#打印不换行
'''
def test(a,*b): 
 print (a,end='') 
 for i in b:
  print (i,end='')
test(10,'c','v')
'''
#输入数字数组
'''
p=input('input some numbers:') #空格隔开的，比如3 4 1 2，会被读成
'3 4 1 2'
p=list(map(int,p.split(' '))) #先用空格分隔
，变成字符型数组，再用map整合成int型数组，python3里map返回的不是list，故需要再调整成list
'''
#序列操作
'''
num=[1,2,3,4,5,6,7,8,9,10]
print(num[1:3]) #左开右闭，结果是[2,3]
print(num[-3:-1]) #左索引不能晚于右索引，结果是[8,9]
print(num[:])
print(num[1:3:1])
print(num[2:0:-1])
print([1,2]+[3,4,5]) #序列相加，结果是[1,2,3,4,5]
print('ab'+'gh') 
print('ab'*5) #序列乘法
tring=[None]*5 #序列初始化
tring.append(4) #末尾增加元素
del tring[1] #删除指定位置元素
print(len(tring)) #序列长度
print(tring)
'''
'''
boil=list('abcde') 序列化
boil[2:]=list('tttt') #分片赋值，结果是['a','b','t','t','t','t','t']
boil[2:2]=list('in') #分片插入，结果是['a','b','i','n','t','t','t','t']
boil[0:2]=[] #分片删除，结果是['i','n','t','t','t','t]
print(boil)
print(boil.count('t')) #元素个数
haha=[1,2,3]
boil.extend(haha) #序列扩展
print(boil) #序列相加后保存，结果是['i','n','t','t','t','t',1,2,3]
print(boil.index('t')) #索引't'，结果是2
boil.insert(2,'s') #插入元素，结果是['i','n','s','t','t','t','t',1,2,3]
tmp=boil.pop(3) #弹出元素
boil.remove(1) #删除第一个指定元素
print(boil) #结果是['i','n','s','t','t','t',2,3]
boil2=boil[:] #分片复制
boil3=boil2.copy() #复制列表
del boil[:] #清空列表
boil2.clear() #清空列表
print(boil2)
'''
'''
num=[1,7,2,5,9]
n=num[:] #分片复制，否则对n排序会影响num
n.sort() #排序
print(num) #[1,7,2,5,9]
print(n) #[1,2,5,7,9]
'''
#字符串操作
'''
print('%010.2f'%3.12159) #0表示用0补齐
print('%10.2f'%3.12159) #用空格补齐
print('%+10.2f'%3.12159) #+表示加上正负号
print('%-10.2f'%3.12159) #-表示对齐
str='abcce' #查找字符串
str.find(bc,0,4) #要找的子串，起始位置，结束位置；最后返回位置，-1表示没找到
mark='++'
new=mark.join(str) #join方法，用指定字符串连接另一字符串，结果是'a++b++c++c++e'，不改变原字符串
srr2='12','e','ert'
new2=mark.join(str2) #结果是'12++e++ert'
field='DO IT NOW'
field.lower().find('It'.lower()) #转换成小写
field.upper().find('It'.upper()) #转换成大写
field.swapcase() #大小写互换
field.replace('O','xx',1) #将'O'替换成'xx‘，且不超过一次
dirs='usr/data/main/' #split方法是join的逆方法
dirs.split('/',2) #最多两次切片分割，不改变原字符串，结果是['','usr','data/main']
str='---a-b-c--'
str.strip('-') #移除头尾指定的字符，结果是'a-b-c'
 
#str.translate(table,[del])
intable='adfest'
outtable='123456'
trantab=str.maketrans(intable,outtable) #生成替换表
st='just do it'
print(st.translate(trantab)) #按替换表一次性替换多个字符  
'''
#字典方法
'''
d1={'name':'Tom','age':12}
list1=[('name','Tom'),('age',12)]
d2=dict(list1) #元组列表转字典
d2['name']='Jerry' #修改字典内容
del d2['age'] #删除字典内容
d2['age']=15 #添加内容
del d2 #删除字典
len(d1) #字典元素个数
print('%(name)s'%d1) #字典格式化输出
d1.clear() #清空字典
d3={'aa':1,'bb':2,'cc':['p','q']}
d4=d3.copy() #字典浅复制
d4['aa']=3 #此处不影响d3['aa']的值
d4['cc'].remove('p') #此处会影响d3中的'cc'的值
#最后d4={'aa':3,'bb','cc':['q']},d3={'aa':1,'bb':2,'cc':['q']}
seq=('name','age')
info=dict.fromkeys(seq) #用给定元组的元素做键创建空值字典
print('%d'%d3.get('aa')) #get方法
d3.get('ff','未指定') #若无查找键则返回'未指定'
print('gg' in d3) #key in dict方法
k=d3.items() #将字典中的键值对变成可遍历的数组
for i in k:
 print (i)
j=d3.keys() #返回一个包含所有键的元组数组，可遍历，可列表化
d3.setdefault('hh','here') #若找不到'hh',则添加其至字典中，若不指定default值'here'，则默认是'None’
print('%s'%d3)
d4={'aa':5,'nn':14}
d3.update(d4) #用d4的内容对d3进行更新，没有的加入，键相同的覆盖旧值
print (d3)
v=d3.values() #返回字典的所有值，可遍历，可列表化，内容可重复
'''
#函数参数定义顺序必须是：必须参数，默认参数，可变参数、关键字参数
#定义函数注意加':'
'''
def test(name,m1=7,m2=8,*p,**kw): #name是必须参数，m1,m2是默认参数，p代表可变参数,kw传入字典，是关键字参数
 print('name:',name,'m1:',m1,'m2:',m2,'p:',*p,'kw:',kw) #此处*p若改为p，则打印元组(1,2,'k')；kw若改为*kw则打印所有键
test('mimi',5,6,1,2,'k',hometown='changsha',love='jerry') #字符串类型的键不用加引号
#更改多个默认参数：只要有一个默认参数采取传入参数名更改值的方法，则其他想更改值的默认参数也必须传入参数名
tuple1=('mimi',5,6,1,2,'k')
dict1={'hometown':'changsha','love':'jerry'}
test(*tuple1,**dict1) #打印效果一样 
'''
#闭包
'''
#例一
def test(*p):
 def cal():
  a=0
  for i in p:
   a+=i
  return a
 return cal #闭包就是大函数内部嵌套定义小函数，小函数调用大函数的数据，最后大函数返回一个小函数
print(test(1,2,3)) #打印函数
cal=test(1,2,3)
print(cal()) #打印值6
'''
'''
#例二
def line_conf(a,b):
 def line(x):
  return a*x+b
 return line #line_conf函数返回一个函数，此处为一条直线一条直线
line1=line_conf(2,3) #line1是2x+3，此时待指定x
print(line1(4)) 输出11
'''
#递归函数&尾递归
'''
def fact(n):
 if n==1:
  return 1
 return n*fact(n-1) #返回的是表达式
print(fact(5))
def fact2(n):
 return fact_item(n,1)
def fact_item(num,product): #product记录中间结果
 if num==1:
  return product
 return fact_item(num-1,num*product) #尾递归：最后返回函数，而非表达式
print(fact2(5))
'''
'''
#filter函数:filter(func,list)
def func(x):
 return x>3
f_list=filter(func,[1,2,3,4,5])
print([item for item in f_list])
#匿名函数
print([item for item in filter(lambda x:x>3,[1,2,3,4,5])])
t=lambda x,y,z:x+y+z
print(t(3,4,5))
'''
#类的定义与使用
'''
class student(object):
 def __init__(self,name,score,home):
  self.n=name
  self.s=score
  self.__h=home #此时__h是不可外部访问的变量
 def info(self):
  print('学生：%s;分数：%d;home: %s'%(self.n,self.s,self.__h))
 def get_home(self):
  return self.__h
 def set_home(self,home):
  self.__h=home  
st1=student('Tim',97,'hongkong')
st1.info()
st1.set_home('changsha') #设置get_home方法后才能修改内部变量__h
print('%s'%st1.get_home()) #直接用st1.__h会显示该类无__h变量
'''
#类的继承（可同时继承多个类，调用的方法按继承顺序搜索）
'''
class animal(object):
 def __init__(self):
  self.a=1
  self.b=2
 def run(self):
  print('animal is running')
 def __private(self): #这是一个私有方法，子类不能调用它，只能基类内部调用
  pass
 def info(self):
  print('%d %d'%(self.a,self.b))
  
class dog(animal):
 def __init__(self):
  self.a=3
  self.b=4
 def run(self):
  print('dog is running')
dog1=dog() #实例化不能忘
dog1.run() #会调用子类的run而非基类的run
dog1.info() #输出3 4
#print(isinstance(dog1,dog)) #判断是否是某个类
#print(dir(dog1)) #输出包含的类、属性等
'''
#多态
'''
def run2times(obj):
 obj.run()
 obj.run()
run2times(animal()) #输出两次animal is running
run2times(dog1) #输出两次dog is running
'''
#判断函数类型
'''
def func():
 pass
print(type(func)==types.FunctionType) 
print(type(abs)==types.BuiltinFunctionType)
print(type(lambda x:x>3)==types.LambdaType)
'''
#类的专有方法
'''
class student(object):
 def __init__(self,name):
  self.n=name
 def __str__(self):
  return('%s'%self.n)
 __repr__=__str__ #该句可实现交互模式下的正常输出
 
st2=student('xiaoming')
print(st2)
class fib(object):
 def __init__(self):
  self.a,self.b=0,1
 def __iter__(self):
  return self
 def __next__(self):
  self.a,self.b=self.b,self.a+self.b
  if self.a>1000:
   raise StopIteration();
  return self.a
  
for i in fib():
 print (i)

class f2(object):
 def __getitem__(self,p):
   a,b=1,1
   for x in range(p):
    a,b=b,a+b
   return a
   
fi=f2()
print(fi[10])
'''

#@staticmethod（不需要表示自身对象的self和自身类的cls参数，就跟使用函数一样）
#@classmethod（不需要self参数，但第一个参数需要是表示自身类的cls参数）
'''
class A(object):
    bar = 1
    def foo(self):
        print 'foo'
 
    @staticmethod
    def static_foo():
        print 'static_foo'
        print A.bar
 
    @classmethod
    def class_foo(cls):
        print 'class_foo'
        print cls.bar
        cls().foo()
 
A.static_foo()
A.class_foo()
'''

#@装饰器
'''
def check_num(func):
    strs = func()
    if strs.isdigit():
        print('输入为数字：{}...'.format(strs))
    else:
        print('输入不是数字：{}...'.format(strs))
 
@check_num
def get_input():
    strs = input('请输入：')
    return strs
'''


#库的使用
#numpy的使用
import numpy as np
'''
a = np.array([[1,2,3], [4,5,6]])
b = np.array([[1.,2.], [3.,4.], [5.,6.]])
print('a.shape = ', a.shape, 'b.shape = ', b.shape)
print('a.dtype = ', a.dtype, 'b.dtype = ', b.dtype)
#结果如下：
# a.shape = (2, 3) b.shape = (3, 2)
# a.dtype = int32 b.dtype = float64
a=a.astype('float64') #改变数组元素类型
b=b.reshape(2,3) #改变数组尺寸形状，reshape里可用-1替换某个参数，省去手动计算
print('a.dtype= ', a.dtype)
print(b)
#结果如下：
#a.dtype = float64
#b=array([[1,2,3],[4,5,6]])
# 切片与索引
a=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
a=array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]])
a[1,2]==6
a[1::2,1::2]==array([[5,7],[13,15]])
a[:,1]==array([1,5,9,13])
a[2:,2:]==array([[10,11],[14,15]])
a[1:3,[0,1,3]]=array([[4,5,7],[8,9,11]])
a[(0,1,1),(0,1,2)]=array([0,5,6])
b=np.array([0,1,2,3,4,5]) 
b[b>3]==array([4,5]) #条件索引
b[np.array([True,False,False,True,False,False])]==array([0,3]) #布尔索引
mask=np.array([1,0,0,1,0,0],dtype=np.bool) #布尔索引
a[mask]=array([0,3])
#ufunc操作，即函数对数组里每一个元素都进行操作
x=np.linspace(1,2*np.pi,3,endpoint=True)
print('x=',x,'\n')
y=np.sin(x)
print('y=',y,'\n')
np.sin(x,x)
print('x=',x,'\n')
#数组比较
x1=np.array([1,2,6])
x2=np.array([2,3,5]) 
print(x1<x2)  #[ True  True False]
print(np.any(x1<x2)) #True
print(np.all(x1<x2)) #False
#矩阵点乘
x1=np.array([[1,2,3],[4,5,6]])
x2=np.array([[3,4],[5,6],[7,8]])
print(np.dot(x1,x2)) 
#以文件形式保存和加载array数组
a=np.array([[1,2,3],[4,5,6]])
np.save('a.npy',a)
c=np.load('a.npy')
 
np.savetxt('a.txt',a)
d=np.loadtxt('a.txt')
'''
'''
#数据归一化，按列减去均值除以方差，每一列的均值归零，方差归一
from sklearn import preprocessing
x=np.array([[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]])
x_scaled=preprocessing.scale(x)
print(x_scaled)
print(x_scaled.mean(axis=0))
print(x_scaled.std(axis=0))
'''
'''
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]
[0. 0. 0.]
[1. 1. 1.]
'''
'''
#主成分分析
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
x=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y=np.array([1,0,0,1,2,2])
pca=PCA(n_components=2) #mle表示自动选取合适的维数
reduced_x=pca.fit_transform(x)
print(reduced_x)
print(pca.explained_variance_ratio_)
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)):
    if y[i] ==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.ylim(-3.0,3.0)
#可视化
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
'''
'''
x=[[4,5],[0,1],[6,7],[2,3],[8,9]]
y=[2,0,3,1,4]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=57)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
'''
#线性回归
'''
# 利用 diabetes数据集来学习线性回归
# diabetes 是一个关于糖尿病的数据集， 该数据集包括442个病人的生理数据及一年以后的病情发展情况。 
# 数据集中的特征值总共10项, 如下:
    # 年龄
    # 性别
    #体质指数
    #血压
    #s1,s2,s3,s4,s4,s6  (六种血清的化验数据)
    #但请注意，以上的数据是经过特殊处理， 10个数据中的每个都做了均值中心化处理，然后又用标准差乘以个体数量调整了数值范围。验证就会发现任何一列的所有数值平方和为1. 
    
#关于数据集更多的信息: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
   # http://scikit-learn.org/stable/datasets/index.html#datasets
   
import numpy as np
from sklearn import datasets
diabetes=datasets.load_diabetes()
#查看第一列年龄的数据
print('=================age================','\n',diabetes.data[:,0])
#求证： 每一列的数值的平方和为1
print('sum=',np.sum( diabetes.data[:,0]**2))  #求年龄列
#糖尿病进展的数据
print(diabetes.target)  #数值介于   25到346之间
#切分训练集与测试集
#自动切分训练集太小了
#from sklearn.cross_validation import train_test_split
#x_train,x_test,y_train,y_test=train_test_split( diabetes.data,diabetes.target,random_state=14)
#所以换成手工切分
x_train=diabetes.data[:-20]
y_train=diabetes.target[:-20]
x_test=diabetes.data[-20:]
y_test=diabetes.target[-20:]
 
#什么是回归呢? 回归的目的是预测数值型的目标值。最直接的办法是根据训练数据计算出一个求目标值的计算公式。假如你想预测一个地区的餐馆数量，可能会这么计算：
#     num = 0.002 * people + 0.001 * gpd
# 以上就是所谓的回归方程，其中的0.002, 0.001称作回归系数，求这些回归系数的过程就是回归。一旦求出了这些回归系数，再给定输入，做预测就简单了. 
# 回归分为线性回归和非线性回归。 上面的公式描述的就是线性回归. 
 
#线性回归通过拟合线性模型的回归系数W =（w_1，…，w_p）来减少数据中观察到的结果和实际结果之间的残差平方和，并通过线性逼近进行预测。
 
#scikit-learn库的线性回归预测模型通过fit(x,y)方法来训xaisaj型，其中x为数据的属性，y为所属的类型.线性模型的回归系数W会保存在它的coef_方法中. 
from sklearn import linear_model
linreg=linear_model.LinearRegression()   #创建线性回归
 
#用训练集训练模型
linreg.fit( x_train,y_train)
#调用预测模型的coef_属性,求出每种生理数据的回归系数b, 一共10个结果，分别对应10个生理特征.
print(linreg.coef_)
 
#在模型上调用predict()函数，传入测试集，得到预测值，
linreg.predict( x_test )
#结果:array([ 197.61846908,  155.43979328,  172.88665147,  111.53537279,
  #      164.80054784,  131.06954875,  259.12237761,  100.47935157,
  #      117.0601052 ,  124.30503555,  218.36632793,   61.19831284,
  #      132.25046751,  120.3332925 ,   52.54458691,  194.03798088,
  #      102.57139702,  123.56604987,  211.0346317 ,   52.60335674])
    
 
#array([ 233.,   91.,  111.,  152.,  120.,   67.,  310.,   94.,  183.,
#         66.,  173.,   72.,   49.,   64.,   48.,  178.,  104.,  132.,
#        220.,   57.])
 
#如何评价以上的模型优劣呢？我们可以引入方差，方差越接近于1,模型越好. 
# 方差: 统计中的方差（样本方差）是各个数据分别与其平均数之差的平方的和的平均数
print(linreg.score( x_test,y_test))
 
 
#对每个特征绘制一个线性回归图表
import matplotlib.pyplot as plt
#matplot显示图例中的中文问题 :   https://www.zhihu.com/question/25404709/answer/67672003
import matplotlib.font_manager as fm
#mac中的字体问题请看: https://zhidao.baidu.com/question/161361596.html
 
 
plt.figure(  figsize=(8,12))
#循环10个特征
for f in range(0,10):
    #取出测试集中第f特征列的值, 这样取出来的数组变成一维的了，
    xi_test=x_test[:,f]
    #取出训练集中第f特征列的值
    xi_train=x_train[:,f]
    
    #将一维数组转为二维的
    xi_test=xi_test[:,np.newaxis]
    xi_train=xi_train[:,np.newaxis]
    
    plt.ylabel(u'病情数值')
    linreg.fit( xi_train,y_train)   #根据第f特征列进行训练
    y=linreg.predict( xi_test )       #根据上面训练的模型进行预测,得到预测结果y
    
    #加入子图
    plt.subplot(5,2,f+1)   # 5表示10个图分为5行, 2表示每行2个图, f+1表示图的编号，可以使用这个编号控制这个图
    #绘制点   代表测试集的数据分布情况
    plt.scatter(  xi_test,y_test,color='k' )
    #绘制线
    plt.plot(xi_test,y,color='b',linewidth=3)
    
plt.savefig('python_糖尿病数据集_预测病情_线性回归_最小平方回归.png')
plt.show()
'''
#爬虫
'''
from lxml import etree
import requests
import time
for a in range(10):
    url = 'https://movie.douban.com/top250?start={}'.format(a*25)
    data = requests.get(url).text
    # print(data)
    s = etree.HTML(data)
    file = s.xpath('//*[@id="content"]/div/div[1]/ol/li')
    for div in file:
        movies_name = div.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
        movies_score = div.xpath('./div/div[2]/div[2]/div/span[2]/text()')[0]
        movies_href = div.xpath('./div/div[2]/div[1]/a/@href')[0]
        movies_number = div.xpath('./div/div[2]/div[2]/div/span[4]/text()')[0].strip("(").strip( ).strip(")")
        movie_scrible = div.xpath('./div/div[2]/div[2]/p[2]/span/text()')
        # time.sleep(1)
        if len(movie_scrible)>0:
            print("{} {} {} {} {}".format(movies_name,movies_href,movies_score,movies_number,movie_scrible[0]))
        else:
            print("{} {} {} {}".format(movies_name,movies_href,movies_score,movies_number))
'''
#用with处理异常
'''
class Test:
  def __enter__(self):
    print('__enter__() is call!')
    return self
  def dosomething(self):
    x = 1/0
    print('dosomethong!')
  def __exit__(self, exc_type, exc_value, traceback):
    print('__exit__() is call!')
    print(f'type:{exc_type}')
    print(f'value:{exc_value}')
    print(f'trace:{traceback}')
    print('__exit()__ finished')
    return True
 
with Test() as sample:
  sample.dosomething()
'''

#打印不换行
'''
def test(a,*b): 
 print (a,end='') 
 for i in b:
  print (i,end='')
test(10,'c','v')
'''
#序列操作
'''
num=[1,2,3,4,5,6,7,8,9,10]
print(num[1:3]) #左开右闭，结果是[2,3]
print(num[-3:-1]) #左索引不能晚于右索引，结果是[8,9]
print(num[:])
print(num[1:3:1])
print(num[2:0:-1])
print([1,2]+[3,4,5]) #序列相加，结果是[1,2,3,4,5]
print('ab'+'gh') 
print('ab'*5) #序列乘法
tring=[None]*5 #序列初始化
tring.append(4) #末尾增加元素
del tring[1] #删除指定位置元素
print(len(tring)) #序列长度
print(tring)
'''
'''
boil=list('abcde') #字符串序列化
boil[2:]=list('tttt') #分片赋值，结果是['a','b','t','t','t','t','t']
boil[2:2]=list('in') #分片插入，结果是['a','b','i','n','t','t','t','t']
boil[0:2]=[] #分片删除，结果是['i','n','t','t','t','t]
print(boil)
print(boil.count('t')) #元素个数
haha=[1,2,3]
boil.extend(haha) #序列扩展
print(boil) #序列相加后保存，结果是['i','n','t','t','t','t',1,2,3]
print(boil.index('t')) #索引't'，结果是2
boil.insert(2,'s') #插入元素，结果是['i','n','s','t','t','t','t',1,2,3]
tmp=boil.pop(3) #弹出元素
boil.remove(1) #删除第一个指定元素
print(boil) #结果是['i','n','s','t','t','t',2,3]
boil2=boil[:] #分片复制
boil3=boil2.copy() #复制列表
del boil[:] #清空列表
boil2.clear() #清空列表
print(boil2)
'''
'''
num=[1,7,2,5,9]
n=num[:] #分片复制，否则对n排序会影响num
n.sort() #排序
print(num) #[1,7,2,5,9]
print(n) #[1,2,5,7,9]
'''


#字符串操作
'''
print('%010.2f'%3.12159) #0表示用0补齐
print('%10.2f'%3.12159) #用空格补齐
print('%+10.2f'%3.12159) #+表示加上正负号
print('%-10.2f'%3.12159) #-表示对齐
str='abcce' #查找字符串
str.find(bc,0,4) #要找的子串，起始位置，结束位置；最后返回位置，-1表示没找到
mark='++'
new=mark.join(str) #用指定字符串连接另一字符串，结果是'a++b++c++c++e'
srr2='12','e','ert'
new2=mark.join(str2) #结果是'12++e++ert'
field='DO IT NOW'
field.lower().find('It'.lower()) #转换成小写
field.upper().find('It'.upper()) #转换成大写
field.swapcase() #大小写互换
field.replace('O','xx',1) #将'O'替换成'xx‘，且不超过一次
'''
#装饰器
'''
import time
def timeit(func):
    def wrapper(): ##内嵌函数，用于包装func
        start = time.clock()
        func()
        end =time.clock()
        print ('used:', end - start)
    return wrapper ##返回包装后的func
 
@timeit ##此处等价于timeit(foo())
def foo():
    print ('some words')

foo()
'''

'''
def makebold(fn):
    def wrapped():
        return ("<b>" + fn() + "</b>")
    return wrapped
def makeitalic(fn):
    def wrapped():
        return ("<i>" + fn() + "</i>")
    return wrapped
@makebold
@makeitalic
def hello():
    return ("hello world")
print (hello()) ## 返回 <b><i>hello world</i></b>
'''

'''
#python命令行传参：
#方法一：sys.argv
#aa.py内容如下
import sys
print sys.argv[0]
print sys.argv[1]
print sys.argv[2]
print sys.argv[3]

#命令行 
python argv.py 1 2 3

#结果
aa.py
#1
#2
#3

#方法二：argparse
#cc.py内容如下
# -*- coding: utf-8 -*-
 
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--integer', type=int, default=3, help='display an integer')
parser.add_argument('-s', '--string', type=str, default='haha', help='display an string')

args = vars(parser.parse_args()) #vars将之变成字典
 
print (args['integer'])
print (args['string'])

if __name__ == '__main__' :	
	print(parser.parse_args())


#命令行
python cc.py -i 10 --string 'heihei'

#结果
#10
#'heihei'
'''

#import同文件夹下文件
#aa.py内容如下：
from sys import argv

def aa(a, b):
	print(a + b)
	
class dd(object):
	att = 1
	def __init__(self, name, age):
		self.name = name
		self.age = age
	def info(self):
		print(self.name, self.age, 'info')
	def __call__(self, t):
		print('age:', self.age + t)
	@classmethod
	def haha(cls, lenth):
		print(lenth)
		print(cls.att)
	
	#类方法可用于新建init函数
	@classmethod
	def reinit(cls, string, num):
		name = string + string
		age = num * 2
		dd1 =  cls(name, age)
		return dd1 #返回一个dd类的实例
	
	@staticmethod
	def hoho(weight):
		print(weight)

dic = {
	'model' : 'bidaf',
	'num' : 100,
	'end' : 11
}

lili = [1, 2, 3]

d = dd.reinit('xiao', 6)
d.info()

if __name__ == '__main__':
	print(argv[0])

#bb.py内容如下：
from aa import dic
from aa import dd
from aa import lili
print(dic['model'])
print(lili[1])
dd.hoho(50)
dd.haha(30)

#import同文件夹下子文件夹
#需要在子文件夹中加入__inti__.py（内容空白即可）才能使用子文件夹中的文件

#引入本地包
导入包的语句之前添加下列语句
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

#获取当前路径
import os
print ('***获取当前文件名***')
print (__file__)

print ('***获取当前目录***')
print (os.getcwd())
print (os.path.abspath(os.path.dirname(__file__)))

print ('***获取上级目录***')
print (os.path.abspath(os.path.dirname(os.getcwd())))
print (os.path.abspath(os.path.join(os.getcwd(), "..")))

print ('***获取上上级目录***')
print (os.path.abspath(os.path.join(os.getcwd(), "../..")))

