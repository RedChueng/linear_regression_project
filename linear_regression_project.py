
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[3]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
def unit_matrix(r):
    m = [[0]*r for row in range(r)]#初始化r行r列的0矩阵
    for i in range(r):
        m[i][i] = 1
    return m

I = unit_matrix(4)
# print I


# ## 1.2 返回矩阵的行数和列数

# In[4]:


# TODO 返回矩阵的行数和列数
def shape(M):
    return len(M),len(M[0])
# print shape(I)


# ## 1.3 每个元素四舍五入到特定小数数位

# In[6]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for e in M:
        for i in range(len(e)):
            e[i] = round(e[i], decPts)
#     return M
    pass
# matxRound(I)
# print I


# ## 1.4 计算矩阵的转置

# In[7]:


# TODO 计算矩阵的转置
# 把M看作是由多个list zipped而成，使用*zipped解压，完成行列变换
# 使用map将tuple转成list
def transpose(M):
    return map(list, zip(*M))
# print transpose(A)


# ## 1.5 计算矩阵乘法 AB

# In[8]:


# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    ra = shape(A)[0]
    ca = shape(A)[1]
    rb = shape(B)[0]
    cb = shape(B)[1]
    if ca == rb:
        m = [[0]*cb for row in range(ra)]#初始化row(A)col(B)的0矩阵
        for i in range(ra):
            for j in range(cb):
                for k in range(ca):
                    m[i][j] += A[i][k]*B[k][j]
        return m
    else:
        pass

# print matxMultiply(A, B)


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[9]:


import pprint

pp = pprint.PrettyPrinter(width=20)

#TODO 测试1.2 返回矩阵的行和列

print '测试1.2 返回矩阵的行和列---------------------\n'
pp.pprint(B)
print '\n'
print (shape(B))
print '\n'

#TODO 测试1.3 每个元素四舍五入到特定小数数位

print '测试1.3 每个元素四舍五入到特定小数数位---------------------\n'
C = [[1.0001,2.000002,3.000003,4.04040404,5.55555555,6.1616161616]]
pp.pprint(C)
matxRound(C)
print '\n'
pp.pprint(C)
print '\n'

#TODO 测试1.4 计算矩阵的转置

print '测试1.4 计算矩阵的转置-----------------------\n'
pp.pprint(B)
print '\n'
pp.pprint (transpose(B))
print '\n'

#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘

print '测试1.5 计算矩阵乘法AB，AB无法相乘-----------------------\n'
B = transpose(B)
pp.pprint(A)
print '\n'
pp.pprint(B)
print '\n'
pp.pprint (matxMultiply(A,B))
print '\n'

#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘

print '测试1.5 计算矩阵乘法AB，AB可以相乘-----------------------\n'
B = transpose(B)
print '\n'
pp.pprint(A)
print '\n'
pp.pprint(B)
print '\n'
pp.pprint (matxMultiply(A,B))


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[10]:


# TODO 构造增广矩阵，假设A，b行数相同
# 为了避免原矩阵的改变，使用deepcopy方法复制一个相同的矩阵
from copy import deepcopy

def augmentMatrix(A, b):
    a = deepcopy(A)
    for i in range(len(a)):
        a[i].append(b[i][0])
    return a


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[15]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]
    pass

# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        for i in range(len(M[r])):
            M[r][i] = M[r][i]*scale
    pass

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    for i in range(len(M[r1])):
        M[r1][i] += M[r2][i] * scale
    pass


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[13]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    res = [[0] for row in A]  # 用于存储计算结果
    rA = shape(A)[0]
    rb = shape(b)[0]
    if rA != rb: # 检查A，b是否行数相同
        return None
    else:
        s = 0 # 用于储存忽略行变换的次数，此次数与行数相等时，说明得到了结果
        Ab = augmentMatrix(A, b) # 构造增广矩阵Ab
        j = 0
        while j < (shape(Ab)[1]):
            c = [] # 用于储存当前列主对角线及以下元素
            for i in range(j, shape(Ab)[0]):
                c.append(abs(Ab[i][j])) # 对元素取绝对值
            # print '第{0}列主对角线及以下元素绝对值的列表为{1}'.format(j, c)
            m = max(c) # 获得c中的最大值
            # print '第{0}列的最大值为{1}'.format(j, m)
            # print 'm:', m
            # print 'i:', i 
            # print 'j:', j
            r = c.index(max(c)) + j # 获得绝对值最大的元素所在的行
            if m <= epsilon:
                return None
            elif m == 1.0 and r == j:
                s += 1
                j += 1
                # print 's+1'
                pass
            else:
                # print '第{0}列最大值所在的行为{1}'.format(j, r)
                swapRows(Ab, j, r) # 使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c）
                # pp.pprint(Ab)
                scale = 1.0/Ab[j][j] # 第j行第j列元素的乘子
                # print '第二个行变换的scale是：', scale
                scaleRow(Ab, j, scale) # 使用第二个行变换，将列c的对角线元素缩放为1
                # print '第二个行变换的结果是：'
                # pp.pprint(Ab)
                for k in range(shape(Ab)[0]): # 多次使用第三个行变换，将列c的其他元素消为0
                    if k == j:
                        pass
                    else:
                        addScaledRow(Ab, k, j, -Ab[k][j]) # 使用第三个行变换，将列c的其他元素消为0
                        # print '第{0}次使用第三个行变换'.format(k)
                        # pp.pprint(Ab)
                j = 0
                s = 0
            # print 's:', s
            if s == shape(Ab)[0]:
                break

        matxRound(Ab, decPts) # 对增广矩阵元素进行四舍五入到特定小数位
        resc = shape(Ab)[0]
        for x in range(shape(Ab)[0]):
            res[x][0] += Ab[x][resc]
        return res
    
a = [[1,3,1],[2,1,1],[2,2,1]]
b = [[11],[8],[10]]

print gj_Solve(a,b)


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：
# 
# $ A^{T} = \begin{bmatrix}
# I & Z^{T} \\ 
# X^{T} & Y^{T}
# \end{bmatrix} $
# 
# $ \left | A^{T} \right | = \left | I \right |\ast \left | Y^{T} \right | - \left | X^{T} \right | * \left | Z^{T} \right | $
# 
# $ \because  \text{Y 的第一列全0} $
# 
# $ \therefore Y^{T} \text{的第一行全0} $
# 
# $ \text{根据行列式的性质：如果行列式中有一行元素全为零,则行列式的值为零} $
# 
# $ \therefore \left | Y^{T} \right | = 0 $
# 
# $ \because I \text{是单位矩阵} $
# 
# $ \therefore \left | I \right | = 1 $
# 
# $ \because \text{Z 为全0矩阵} $
# 
# $ \therefore \left | Z^{T} \right | = 0 $
# 
# $ \text{综上所述} $
# 
# $ \left | A^{T} \right | = 0 $
# 
# $ \because \left | A \right | = \left | A^{T} \right | $
# 
# $ \therefore \left | A \right | = 0 $
# 
# $ \therefore \text{A为奇异矩阵} $

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[19]:


# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
# TODO 求解 x 使得 Ax = b
# TODO 计算 Ax
# TODO 比较 Ax 与 b

print '测试奇异矩阵的结果---------------------\n'
A = [[1,0,4],[0,1,0],[0,0,0]]
b = [[1],[2],[3]]

pp.pprint(A)
print '\n'
pp.pprint(b)
print '\n'
print gj_Solve(A, b)
print '\n'

print '测试非奇异矩阵的结果---------------------\n'
A = [[1,3,1],[2,1,1],[2,2,1]]
b = [[11],[8],[10]]

pp.pprint(A)
print '\n'
pp.pprint(b)
print '\n'

print '求解 x 使得 Ax = b\n'

x = gj_Solve(A, b)

pp.pprint(x)
print '\n'
print '计算 Ax \n'

Ax = matxMultiply(A, x)

pp.pprint(Ax)
print '\n'

print '比较 Ax 与 b： \n'

def compare(A, B):
    if A == B:
        return True
    else:
        return False

print compare(Ax,b)


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# 
# $$
# \frac{\partial E}{\partial m} = 2\sum_{i = 1}^{n}(y_{i}-mx_{i}-b)\frac{\partial (y_{i}-mx_{i}-b)}{\partial m}
# $$
# $$
# =2\sum_{i = 1}^{n}(y_{i}-mx_{i}-b)(-x_{i})
# $$
# $$
# =\sum_{i = 1}^{n}-2x_{i}(y_{i}-mx_{i}-b)
# $$
# 
# $$
# \frac{\partial E}{\partial b} = 2\sum_{i = 1}^{n}(y_{i}-mx_{i}-b)\frac{\partial (y_{i}-mx_{i}-b)}{\partial b}
# $$
# $$
# =2\sum_{i = 1}^{n}(y_{i}-mx_{i}-b)(-1)
# $$
# $$
# =\sum_{i = 1}^{n}-2(y_{i}-mx_{i}-b)
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 
# \begin{bmatrix}
# \sum_{i = 1}^{n}-2x_{i}(y_{i}-mx_{i}-b) \\ 
# \sum_{i = 1}^{n}-2(y_{i}-mx_{i}-b)
# \end{bmatrix}
# $$
# 
# $$
# 2X^TXh - 2X^TY =
# 2X^T(Xh - Y)
# $$
# 
# $$
# (Xh - Y) = 
# \begin{bmatrix}
# mx_{1} + b - y_{_{1}} \\ 
# mx_{2} + b - y_{_{2}} \\ 
# \cdots \\ 
# mx_{n} + b - y_{_{n}} 
# \end{bmatrix}
# $$
# 
# $$
# 2X^T(Xh - Y) = 
# 2\begin{bmatrix}
# \sum_{i=1}^{n}(mx_{i} + b - y_{_{i}})x_{i} \\ 
# \sum_{i=1}^{n}(mx_{i} + b - y_{_{i}}) \\ 
# \end{bmatrix}
# $$
# 
# $$
# =
# \begin{bmatrix}
# \sum_{i = 1}^{n}-2x_{i}(y_{i}-mx_{i}-b) \\ 
# \sum_{i = 1}^{n}-2(y_{i}-mx_{i}-b)
# \end{bmatrix}
# $$
# 
# $$
# = 
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}
# $$
# 
# $$
# \text{即：}
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[21]:


# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
# 把方程看成AX=b的形式，利用gj_Solve函数求解
def linearRegression(points):
    X = [[points[i][0],1] for i in range(len(points))]
    Y = [[points[i][1]]for i in range(len(points))]

    XT = transpose(X)

    A = matxMultiply(XT, X)
    b = matxMultiply(XT, Y)

    h = gj_Solve(A, b)

    m = h[0]
    b = h[1]
    return m, b


# ## 3.3 测试你的线性回归实现

# In[22]:


# TODO 构造线性函数

m = 2
b = 4

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random

gs = [random.gauss(0,1) for i in range(100)] # 产生100个服从标准正态分布的数
x = [random.uniform(-10,10) for i in range(100)]
y = [m*x[i] + b + gs[i] for i in range(100)] # y值加上高斯噪音
points = map(list,zip(x, y))

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较

m_g = linearRegression(points)[0]
b_g = linearRegression(points)[1]

print m_g, b_g


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[16]:


import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))

            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[ ]:




