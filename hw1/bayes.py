import numpy as np
import numpy.linalg as la
import argparse
import csv
np.version.version
train_data 	= np.genfromtxt('propublicaTrain.csv',usecols=(1,2,3,4,5,6,7,8,9),skip_header = 1, dtype = 'int8',delimiter=',')
train_label	= np.genfromtxt('propublicaTrain.csv', usecols=(0), skip_header = 1, dtype = 'int8',delimiter=',')

test_data 	= np.genfromtxt('propublicaTest.csv',usecols=(1,2,3,4,5,6,7,8,9),skip_header = 1, dtype = 'int8',delimiter=',')
test_label 	= np.genfromtxt('propublicaTest.csv', usecols=(0), skip_header = 1, dtype = 'int8',delimiter=',')
train_num,_ = train_data.shape
test_num,_ = test_data.shape

condition_0 = np.equal(train_label,np.zeros(train_num,dtype='int8'))
data_0 = train_data[condition_0]

condition_1 = np.equal(train_label,np.ones(train_num, dtype='int8'))
data_1 = train_data[condition_1]
print(data_0.shape)
print(data_1.shape)

num_0,_ = data_0.shape
num_1,_ = data_1.shape

prob_0 = num_0/train_num
prob_1 = num_1/train_num

mean_0 = np.mean(data_0, axis = 0)
mean_1 = np.mean(data_1, axis = 0)

cov_0 = np.cov(np.transpose(data_0))
cov_1 = np.cov(np.transpose(data_1))

small = np.eye(cov_0.shape[0]) * 0.000001
cov_inv_0 = la.inv(cov_0+small)

small = np.eye(cov_1.shape[0]) * 0.000001
cov_inv_1 = la.inv(cov_1+small)

count = 0
for i in range(train_num):
	val = train_data[i]
	val_0 = val - mean_0
	val_1 = val - mean_1
	result_0 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_0),cov_inv_0),val_0))
	result_1 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_1),cov_inv_1),val_1))
	result_0 *= prob_0
	result_1 *= prob_1
	label = 0
	if result_1>result_0:
		label = 1
	if label == train_label[i].item(0):
		count+=1
train_acc = count/train_num
print(count/train_num)


count = 0
for i in range(test_num):
	val = test_data[i]
	val_0 = val - mean_0
	val_1 = val - mean_1
	result_0 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_0),cov_inv_0),val_0))
	result_1 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_1),cov_inv_1),val_1))
	result_0 *= prob_0
	result_1 *= prob_1
	label = 0
	if result_1>result_0:
		label = 1
	if label == test_label[i].item(0):
		count+=1
print(count/test_num)
test_acc = count/test_num

with open('result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['bayes',train_acc,test_acc])





