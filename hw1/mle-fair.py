import numpy as np
from numpy.linalg import inv
import csv
np.version.version
train_data 	= np.genfromtxt('propublicaTrain.csv', skip_header = 1, dtype = 'int8',delimiter=',')

test_data 	= np.genfromtxt('propublicaTest.csv', skip_header = 1, dtype = 'int8',delimiter=',')
test_data_0 = test_data[test_data[:,3] == 0]
test_data_1 = test_data[test_data[:,3] == 1]
print(test_data_0.shape)
print(test_data_1.shape)



mean = np.mean(train_data, axis = 0)
var = np.var(train_data, axis = 0)
cov = np.cov(np.transpose(train_data))

m,n = train_data.shape
small = np.eye(cov.shape[0]) *0.00001
cov_inv = inv(cov+small)

m,n = test_data_0.shape
count_0 = [0,0,0]
for i in range(m):
	val = test_data_0[i][1:]
	label_1 = np.array([1])
	label_0 = np.array([0])
	x_1 = np.hstack((label_1,val))
	x_0 = np.hstack((label_0,val))
	res_1 = x_1-mean
	res_0 = x_0-mean
	result_1 = np.exp(-0.5*np.dot(np.dot(np.transpose(res_1),cov_inv),res_1))
	result_0 = np.exp(-0.5*np.dot(np.dot(np.transpose(res_0),cov_inv),res_0))
	# print(type(result_0))
	predict_label = 0
	if result_1.item(0)>result_0.item(0):
		predict_label = 1

	if predict_label==0:
		count_0[0]+=1
	if test_data[i][0].item(0)==0:
		count_0[2]+=1
	if test_data[i][0].item(0)==0 and predict_label==0:
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
	val = test_data_1[i][1:]
	label_1 = np.array([1])
	label_0 = np.array([0])
	x_1 = np.hstack((label_1,val))
	x_0 = np.hstack((label_0,val))
	res_1 = x_1-mean
	res_0 = x_0-mean
	result_1 = np.exp(-0.5*np.dot(np.dot(np.transpose(res_1),cov_inv),res_1))
	result_0 = np.exp(-0.5*np.dot(np.dot(np.transpose(res_0),cov_inv),res_0))
	# print(type(result_0))
	predict_label = 0
	if result_1.item(0)>result_0.item(0):
		predict_label = 1

	if predict_label==0:
		count_1[0]+=1
	if test_data[i][0].item(0)==0:
		count_1[2]+=1
	if test_data[i][0].item(0)==0 and predict_label==0:
		count_1[1]+=1
DP_1 = count_1[0]/m
EO_1 = count_1[1]/count_1[2]
PP_1 = count_1[1]/count_1[0]
print("DP a=1:",DP_1)
print("EO a=1:",EO_1)
print("PP a=1:",PP_1)
print(count_1)

DP_fair = abs(DP_1-DP_0)
EO_fair = abs(EO_1-EO_0)
PP_fair = abs(PP_1-PP_0)
print("DP_fair: ",DP_fair)
print("EO_fair: ",EO_fair)
print("PP_fair: ",PP_fair)
with open('DP_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['mle',DP_fair])

with open('EO_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['mle',EO_fair])

with open('PP_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['mle',PP_fair])
