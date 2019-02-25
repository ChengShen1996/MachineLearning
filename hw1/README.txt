Group member:
Cheng Shen 	cs3750
Qidong Yang qy2216
Jerry Lin 	sl4299

Description:

For classifier, we implemented each classfier in seperated python file.
mle.py: 	Maximum likelihood estimation classifier. 
			python3 mle.py will print out the result

bayes.py:	Naive-bayes classifier.
			python3 bayes.py will print out the result

knn.py:		K nearest neighbor classifier.
			python3 bayes.py -k 3 -n 2 will print out the result
			-k argument is the number of neighbor
			-n argument is the norm option. n is negative, we will use inf norm

plot.py		Helper function
			python3 plot.py will plot the graph needed

mle-fair.py	Measure fairness on MLE classifier
			
bayes-fair.py Measure fairness on Bayes classifer

knn-fair.py	 Measure fairness on knn classifer. k=3 n=2
