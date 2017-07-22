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

a = [['a11','a12'],['a21','a22']]
# b = [['b1'],['b2']]


# print augmentMatrix(a,b)
# print a  
# print b 

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


pp.pprint(A)
addScaledRow(A, 0, 1 ,3)
pp.pprint(A)

# TODO 实现 Gaussain Jordan 方法求解 Ax = b
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
		# print s
		return res
		


a = [[1,3,1],[2,1,1],[2,2,1]]
b = [[11],[8],[10]]

print gj_Solve(a,b)



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

print '比较 Ax 与 b \n'

def compare(A, B):
	if A == B:
		return True
	else:
		return False

print compare(Ax,b)