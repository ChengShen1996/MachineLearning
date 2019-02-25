import pandas as pd
import matplotlib.pyplot as plt
# temp = pd.read_csv("result2.csv", names=['classifier','train_acc','test_acc'])
# temp.plot(x='classifier',y=['train_acc','test_acc'],kind='bar')
# plt.show()



temp = pd.read_csv("PP_result.csv", names=['classifier','fairness'])
temp.plot(x='classifier',y='fairness',kind='bar')
plt.show()

