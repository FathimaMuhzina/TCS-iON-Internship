import pandas as pd
import pickle

df=pd.read_csv('ready_to_model.csv')
df=df.drop(['Unnamed: 0'], axis=1)
#Separating X and y
X=df.drop(['price_range'], axis=1)
y=df['price_range']

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = pd.DataFrame(sc.fit_transform(X),columns = X.columns)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(random_state=42)
dt.fit(X,y)


pickle.dump(dt,open('model.pkl','wb'))
