import numpy as np
import numpy.linalg as la
import argparse
import csv
np.version.version
train_data 	= np.genfromtxt('propublicaTrain.csv', usecols=(1,2,3,4,5,6,7,8,9),skip_header = 1, dtype = 'int8',delimiter=',')
train_label	= np.genfromtxt('propublicaTrain.csv', usecols=(0), skip_header = 1, dtype = 'int8',delimiter=',')

test_data 	= np.genfromtxt('propublicaTest.csv', usecols=(1,2,3,4,5,6,7,8,9),skip_header = 1, dtype = 'int8',delimiter=',')
test_label 	= np.genfromtxt('propublicaTest.csv', usecols=(0), skip_header = 1, dtype = 'int8',delimiter=',')

parser = argparse.ArgumentParser(description = 'knn classifier')
parser.add_argument('-k', '--k', type = int, default =3, help = "Hyper parameter k")
parser.add_argument('-n', '--norm', type = int, default =2, help = "Hyper parameter norm")
args = parser.parse_args()



# print(train_data.shape,train_label.shape, test_data.shape,test_label.shape)
k=args.k
norm_order = args.norm
half = int(k/2)


m,n = train_data.shape
count = 0
for i in range(m):
	diff = np.subtract(train_data,train_data[i])
	norm = la.norm(diff, ord = norm_order, axis = 1)
	temp = np.argsort(norm)
	temp = temp[0:k]

	count_1 = 0
	for x in temp:
		if train_label[x].item(0)>0:
			count_1+=1
	if count_1> half:
		if train_label[i].item(0) ==1:
			count+=1
	else:
		if train_label[i].item(0) == 0:
			count+=1

print('train  ',count/m)
train_acc = count/m

m,n = test_data.shape
count = 0
for i in range(m):
	diff = np.subtract(test_data,test_data[i])
	norm = la.norm(diff, ord = norm_order, axis = 1)
	temp = np.argsort(norm)
	temp = temp[0:k]

	count_1 = 0
	for x in temp:
		if test_label[x].item(0)>0:
			count_1+=1
	if count_1> half:
		if test_label[i].item(0) ==1:
			count+=1
	else:
		if test_label[i].item(0) == 0:
			count+=1

print('test   ',count/m)
test_acc = count/m

with open('result.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(['knn-k'+str(k)+'-c'+str(norm_order),train_acc,test_acc])