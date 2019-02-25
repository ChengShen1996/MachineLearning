import numpy as np
from numpy.linalg import inv
import csv
np.version.version
train_data 	= np.genfromtxt('propublicaTrain.csv', skip_header = 1, dtype = 'int8',delimiter=',')
train_data = train_data[np.random.choice(train_data.shape[0],2000,replace=False)]
print(train_data.shape)

test_data 	= np.genfromtxt('propublicaTest.csv', skip_header = 1, dtype = 'int8',delimiter=',')


print(train_data.shape)
print(test_data.shape)
print(train_data[0])
print(train_data[1])

mean = np.mean(train_data, axis = 0)
var = np.var(train_data, axis = 0)
cov = np.cov(np.transpose(train_data))
# print(mean)
# print(var)
# print(cov.shape)
m,n = train_data.shape
small = np.eye(cov.shape[0]) *0.00001
cov_inv = inv(cov+small)
train_predict = np.zeros(m)
count = 0
for i in range(m):
	val = train_data[i][1:]
	label_1 = np.array([1])
	label_0 = np.array([0])
	x_1 = np.hstack((label_1,val))
	x_0 = np.hstack((label_0,val))
	res_1 = x_1-mean
	res_0 = x_0-mean
	result_1 = np.dot(np.dot(np.transpose(res_1),cov_inv),res_1)
	result_0 = np.dot(np.dot(np.transpose(res_0),cov_inv),res_0)
	# print(type(result_0))
	if result_1.item(0)<result_0.item(0):
		train_predict[i] = 1
	if train_predict[i] == train_data[i][0]:
		count+=1
train_acc = count/m
print("Train acc: ", train_acc)

m,n = test_data.shape
test_predict = np.zeros(m)
count = 0
for i in range(m):
	val = test_data[i][1:]
	label_1 = np.array([1])
	label_0 = np.array([0])
	x_1 = np.hstack((label_1,val))
	x_0 = np.hstack((label_0,val))
	res_1 = x_1-mean
	res_0 = x_0-mean
	result_1 = np.dot(np.dot(np.transpose(res_1),cov_inv),res_1)
	result_0 = np.dot(np.dot(np.transpose(res_0),cov_inv),res_0)
	# print(type(result_0))
	if result_1.item(0)<result_0.item(0):
		
		test_predict[i] = 1
	if test_predict[i] == test_data[i][0]:
		count+=1
test_acc = count/m
print("Test acc: ",test_acc)

with open('result2.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['mle',train_acc,test_acc])
