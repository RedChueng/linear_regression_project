# -*- coding:UTF-8 -*-

import random
import pprint
from copy import deepcopy

pp = pprint.PrettyPrinter(width=20)

A = [[1,2,3],
	[2,3,3],
	[1,2,5]]

B = [[1,2,3,5],
	[2,3,3,5],
	[1,2,5,1]]

#生成一个r行c列矩阵
def random_matrix(r,c):
	m = [[0]*c for row in range(r)]#初始化r行r列的0矩阵
	for e in m:
		for i in range(len(e)):
			e[i] = random.randint(1,10)
	return m

m1 = random_matrix(2,3)
m2 = random_matrix(3,4)
m3 = random_matrix(4,4)

print 'm1 = ',m1
print 'm2 = ',m2
print 'm3 = ',m3

# 创建一个4*4的单位矩阵
def unit_matrix(r):
	m = [[0]*r for row in range(r)]#初始化r行r列的0矩阵
	for i in range(r):
		m[i][i] = 1
	return m

I =  unit_matrix(4)

print I

# 返回矩阵的行数和列数
def shape(M):
	return len(M),len(M[0])


#矩阵中每个元素四舍五入到特定小数数位
def matxRound(M, decPts=4):
	for e in M:
		for i in range(len(e)):
			e[i] = round(e[i], decPts)
	# return M
	pass

# print matxRound(I, decPts=4)

a = [1,2,3]
b = [4,5,6]
zipped = zip(a,b)
print zipped
print zip(*zipped)

#计算矩阵的转置
def transpose(M):
	return map(list, zip(*M))
	return 

print transpose(m1)

#计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
	ra = shape(A)[0]
	ca = shape(A)[1]
	rb = shape(B)[0]
	cb = shape(B)[1]
	if ca == rb:
		m = [[0]*cb for row in range(ra)]
		for i in range(ra):
			for j in range(cb):
				for k in range(ca):
					m[i][j] += A[i][k]*B[k][j]
		return m
	else:
		pass

print 'm1m2 = ', matxMultiply(m1,m2)
print 'm1m3 = ', matxMultiply(m1,m3)


print '测试1.2 返回矩阵的行和列---------------------'
pp.pprint(B)
print (shape(B))


print '测试1.3 每个元素四舍五入到特定小数数位---------------------'
C = [[1.0001,2.000002,3.000003,4.04040404,5.55555555,6.1616161616]]
pp.pprint(C)
matxRound(C)
pp.pprint(C)


print '测试1.4 计算矩阵的转置-----------------------'
pp.pprint(B)
pp.pprint (transpose(B))


print '测试1.5 计算矩阵乘法AB，AB无法相乘-----------------------'
B = transpose(B)
pp.pprint(A)
pp.pprint(B)
pp.pprint (matxMultiply(A,B))


print '测试1.5 计算矩阵乘法AB，AB可以相乘-----------------------'
B = transpose(B)
pp.pprint(A)
pp.pprint(B)
pp.pprint (matxMultiply(A,B))


#构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
	a = deepcopy(A)
	for i in range(len(a)):
		a[i].append(b[i][0])
	return a

# a = [['a11','a12'],['a21','a22']]
# b = [['b1'],['b2']]


# print augmentMatrix(a,b)
# print a  
# print b 

# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值

def swapRows(M, r1, r2):
	
	pass