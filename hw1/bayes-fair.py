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
test_data_0 = test_data[test_data[:,2] == 0]
test_data_1 = test_data[test_data[:,2] == 1]
print(test_data_0.shape)
print(test_data_1.shape)



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


m,n = test_data_0.shape
count_0 = [0,0,0]
for i in range(m):
	val = test_data_0[i]
	val_0 = val - mean_0
	val_1 = val - mean_1
	result_0 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_0),cov_inv_0),val_0))
	result_1 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_1),cov_inv_1),val_1))
	result_0 *= prob_0
	result_1 *= prob_1
	predict_label = 0
	if result_1>result_0:
		predict_label = 1
	if predict_label==0:
		count_0[0]+=1
	if test_label[i].item(0) ==0:
		count_0[2]+=1
	if test_label[i].item(0) == 0 and predict_label ==0:
		count_0[1]+=1

DP_0 = count_0[0]/m
EO_0 = count_0[1]/count_0[2]
PP_0 = count_0[1]/count_0[0]
print("DP a=0:",DP_0)
print("EO a=0:",EO_0)
print("PP a=0:",PP_0)
print(count_0)


m,n = test_data_1.shape
count_1 = [0,0,0]
for i in range(m):
	val = test_data_1[i]
	val_0 = val - mean_0
	val_1 = val - mean_1
	result_0 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_0),cov_inv_0),val_0))
	result_1 = np.exp(-0.5*np.dot(np.dot(np.transpose(val_1),cov_inv_1),val_1))
	result_0 *= prob_0
	result_1 *= prob_1
	predict_label = 0
	if result_1>result_0:
		predict_label = 1
	if predict_label==0:
		count_1[0]+=1
	if test_label[i].item(0) ==0:
		count_1[2]+=1
	if test_label[i].item(0) == 0 and predict_label ==0:
		count_1[1]+=1

DP_1 = count_1[0]/m
EO_1 = count_1[1]/count_1[2]
PP_1 = count_1[1]/count_1[0]
print("DP a=0:",DP_1)
print("EO a=0:",EO_1)
print("PP a=0:",PP_1)
print(count_1)


DP_fair = abs(DP_1-DP_0)
EO_fair = abs(EO_1-EO_0)
PP_fair = abs(PP_1-PP_0)
print("DP_fair: ",DP_fair)
print("EO_fair: ",EO_fair)
print("PP_fair: ",PP_fair)


with open('DP_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['bayes',DP_fair])

with open('EO_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['bayes',EO_fair])

with open('PP_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['bayes',PP_fair])

