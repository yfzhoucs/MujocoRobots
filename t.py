def func(x=[], y=[6, 7]):
	x.append(8)
	y.append(8)
	return x + y


a = [1, 2]
b = [3, 4]
t = func(x=a)
t = func(y=b)
res = func()
print(res)


import numpy as np
def func(x=np.array([3, 4]), y=np.array([6, 7])):
	# x = np.append(x, np.array([8]), axis=0)
	# y = np.append(y, np.array([8]), axis=0)
	# return np.concatenate((x, y), axis=0)
	x[0] += 8
	y[0] += 8
	return np.concatenate((x, y), axis=0)


a = np.array([1, 2])
b = np.array([3, 4])
t = func(x=a)
t = func(y=b)
res = func()
print(res)