from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()

print(data.data)
print(data.target)
print(data.data.shape)
print(data.target.shape)


x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

with open('model.pkl','wb') as f:
    pickle.dump(model,f)