import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("/Users/sanket.padhi/Desktop/Apexon/Machine_learning/Regression/untitled folder/Machine_learning/Linear_Regression/Simple_linear_Regresion/placement.csv")
print(data.head(10))

plt.scatter(data['cgpa'],data['package'])
plt.xlabel('CGPA')
plt.ylabel('PACKAGE')
# plt.show() #will show the scatter plot between cgpa and package .

X = data.iloc[:,0:1]
Y = data.iloc[:,1:2]
print("X is", X)
print("Y is",Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2, random_state=2)

lr = LinearRegression()

lr.fit(X_train, Y_train)

print("X_test is :", X_test)
print("Y_test is :", Y_test)


print(lr.predict(X_test.iloc[0].values.reshape(1,1)))

