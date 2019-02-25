import numpy as np
import numpy.linalg as la
import argparse
import csv
np.version.version
train_data 	= np.genfromtxt('propublicaTrain.csv', usecols=(1,2,3,4,5,6,7,8,9),skip_header = 1, dtype = 'int8',delimiter=',')
train_label	= np.genfromtxt('propublicaTrain.csv', usecols=(0), skip_header = 1, dtype = 'int8',delimiter=',')

test_data 	= np.genfromtxt('propublicaTest.csv', usecols=(1,2,3,4,5,6,7,8,9),skip_header = 1, dtype = 'int8',delimiter=',')
test_label 	= np.genfromtxt('propublicaTest.csv', usecols=(0), skip_header = 1, dtype = 'int8',delimiter=',')

train_num,_ = train_data.shape
test_num,_ = test_data.shape
test_data_0 = test_data[test_data[:,2] == 0]
test_data_1 = test_data[test_data[:,2] == 1]
print(test_data_0.shape)
print(test_data_1.shape)


parser = argparse.ArgumentParser(description = 'knn classifier')
parser.add_argument('-k', '--k', type = int, default =3, help = "Hyper parameter k")
parser.add_argument('-n', '--norm', type = int, default =2, help = "Hyper parameter norm")
args = parser.parse_args()



# print(train_data.shape,train_label.shape, test_data.shape,test_label.shape)
k=args.k
if args.norm<0:
	norm_order = np.inf
else:
	norm_order = args.norm
half = int(k/2)



m,n = test_data_0.shape
count_0 = [0,0,0]
for i in range(m):
	diff = np.subtract(train_data,test_data_0[i])
	norm = la.norm(diff, ord = norm_order, axis = 1)
	temp = np.argsort(norm)
	temp = temp[0:k]

	label_1 = 0
	for x in temp:
		if train_label[x].item(0)>0:
			label_1+=1
	predict_label = 0
	if label_1> half:
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
	diff = np.subtract(train_data,test_data_1[i])
	norm = la.norm(diff, ord = norm_order, axis = 1)
	temp = np.argsort(norm)
	temp = temp[0:k]

	label_1 = 0
	for x in temp:
		if train_label[x].item(0)>0:
			label_1+=1
	predict_label = 0
	if label_1> half:
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
	writer.writerow(['knn',DP_fair])

with open('EO_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['knn',EO_fair])

with open('PP_result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['knn',PP_fair])



