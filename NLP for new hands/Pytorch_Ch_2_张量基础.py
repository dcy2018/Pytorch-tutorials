
# coding: utf-8

# In[55]:


#-----------------2.1张量基础
import torch
a = torch.ones(3)
a
#tensor([1., 1., 1.])

a[1]
#tensor(1.)

float(a[1])
#1.0

a[2] = 2.0
a
#tensor([1., 1., 2.])

# 使用.zeros是获取适当大小的数组的一种方法
points = torch.zeros(6)

# 用所需的值覆盖这些0
points[0] = 1.0
points[1] = 4.0
points[2] = 2.0
points[3] = 1.0
points[4] = 3.0
points[5] = 5.0
points = torch.tensor([1.0, 4.0, 2.0, 1.0, 3.0, 5.0])
points
#tensor([1., 4., 2., 1., 3., 5.])

float(points[0]), float(points[1])
#(1.0, 4.0)
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points

#tensor([[1., 4.],
#        [2., 1.],
#        [3., 5.]])

points.shape
#torch.Size([3, 2])

points = torch.zeros(3, 2)
points
#tensor([[0., 0.],
#        [0., 0.],
#        [0., 0.]])

points = torch.FloatTensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points
#tensor([[1., 4.],
#       [2., 1.],
#        [3., 5.]])

points[0, 1]
#tensor(4.)

points[0]
#tensor([1., 4.])

#--------------------2.2张量与存储
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points.storage()
#1.0
#4.0
#2.0
#1.0
#3.0
#5.0
#[torch.FloatStorage of size 6]

points_storage = points.storage()
points_storage[0]
#1.0

points.storage()[1]
#4.0

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points_storage = points.storage()
points_storage[0] = 2.0
points
#tensor([[2., 4.],
#        [2., 1.],
#        [3., 5.]])

#-----------------2.3尺寸、存储与步长
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
second_point = points[1]
second_point.storage_offset()
#2

second_point.size()
#torch.Size([2])

second_point.shape
#torch.Size([2])

points.stride()
#(2, 1)

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
second_point = points[1]
second_point.size()
#torch.Size([2])

second_point.storage_offset()
#2

second_point.stride()
#(1,)

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
second_point = points[1]
second_point[0] = 10.0
points
#tensor([[ 1.,  4.],
#        [10.,  1.],
#        [ 3.,  5.]])

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points
#tensor([[1., 4.],
#        [2., 1.],
#        [3., 5.]])

points_t = points.t()
points_t
#tensor([[1., 2., 3.],
#        [4., 1., 5.]])

id(points.storage()) == id(points_t.storage())
#True

points.stride()
#(2, 1)

points_t.stride()
#(1, 2)

some_tensor = torch.ones(3, 4, 5)
some_tensor.shape, some_tensor.stride()
#(torch.Size([3, 4, 5]), (20, 5, 1))

some_tensor_t = some_tensor.transpose(0, 2)
some_tensor_t.shape, some_tensor_t.stride()
#(torch.Size([5, 4, 3]), (1, 5, 20))

points.is_contiguous(), points_t.is_contiguous()
#(True, False)

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points_t = points.t()
points_t
#tensor([[1., 2., 3.],
#        [4., 1., 5.]])

points_t.storage()
# 1.0
# 4.0
# 2.0
# 1.0
# 3.0
# 5.0
#[torch.FloatStorage of size 6]

points_t.stride()
#(1, 2)

points_t_cont = points_t.contiguous()
points_t_cont
#tensor([[1., 2., 3.],
#        [4., 1., 5.]])

points_t_cont.stride()
#(3, 1)

points_t_cont.storage()
# 1.0
# 2.0
# 3.0
# 4.0
# 1.0
# 5.0
#[torch.FloatStorage of size 6]

#----------2.4数据类型
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
short_points.dtype
#torch.int16

double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)

points = torch.randn(10, 2)
short_points = points.type(torch.short)


#------------2.5索引张量
some_list = list(range(6))
some_list[:]     # 所有元素
some_list[1:4]   # 第1（含）到第4（不含）个元素
some_list[1:]    # 第1（含）个之后所有元素
some_list[:4]    # 第4（不含）个之前所有元素
some_list[:-1]   # 最末尾（不含）元素之前所有元素
some_list[1:4:2] # 范围1（含）到4（不含），步长为2的元素

points[1:]    # 第1行及之后所有行，（默认）所有列
points[1:, :] # 第1行及之后所有行，所有列
points[1:, 0] # 第1行及之后所有行，仅第0列


#------------2.6与Numpy的互通性
points = torch.ones(3, 4)
points_np = points.numpy()
points_np
#array([[1., 1., 1., 1.],
#       [1., 1., 1., 1.],
#       [1., 1., 1., 1.]], dtype=float32)

points = torch.from_numpy(points_np)


#------------2.7序列号张量
torch.save(points, '../../data/chapter2/ourpoints.t')

with open('../../data/chapter2/ourpoints.t','wb') as f:
    torch.save(points, f)

points = torch.load('../../data/chapter2/ourpoints.t')

with open('../../data/chapter2/ourpoints.t','rb') as f:
    points = torch.load(f)

conda install h5py

import h5py

f = h5py.File('../../data/chapter2/ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()

f = h5py.File('../../data/chapter2/ourpoints.hdf5', 'r')
dset = f['coords']
last_points = dset[1:]

last_points = torch.from_numpy(dset[1:])
f.close()
# last_points = torch.from_numpy(dset[1:]) # 会报错, 因为f已经关了


#----------2.8将张量转移到GPU上运行
points_gpu = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]], device='cuda')

points_gpu = points.to(device='cuda')

points_gpu = points.to(device='cuda:0')

points = 2 * points # 在CPU上做乘法
points_gpu = 2 * points.to(device='cuda') # 在GPU上做乘法

points_gpu = points_gpu + 4

points_gpu = points.cuda() # 默认为GPU0
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()


#----------2.9张量API
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)

a = torch.ones(3, 2)
a_t = a.transpose(0, 1)

a = torch.ones(3, 2)
a.zero_()
a
#tensor([[0., 0.],
#        [0., 0.],
#        [0., 0.]])


