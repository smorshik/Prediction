import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col=0, parse_dates=True)
df.head()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

def age_category(Age):
    if Age < 18:
        return 0**2
    elif Age < 55:
        return 1**2
    else:
        return 2**2
df['Age'] = df['Age'].apply(age_category)
df['Pclass'] = df['Pclass']**2 

print(df.info())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

xtrain, xtest, ytrain, ytest = train_test_split(df[features], df["Fare"], test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#Pclass Sex Age SibSp Parch
jack = [[9, 0, 1, 1, 0]]
jack_scaled = scaler.transform(jack)

prediction = model.predict(jack_scaled)

print("Цена жилья", prediction)

coeff = pd.DataFrame(model.coef_.T, index=features, columns=['Вес (Weight)'])
print(coeff)
print("Базовый уровень (Intercept):", model.intercept_)
print(model.score(xtest, ytest))

'''predictions = model.predict(xtest)

fig, ax = plt.subplots(1, 2)

#(обучающие данные)
ax[0].scatter(xtrain[:, 0], ytrain, color='blue')
ax[0].set_title("Обучение")

#(тест и предсказания)
ax[1].scatter(xtest[:, 0], ytest, color='orange', alpha=0.5)    # Реальные ответы
ax[1].scatter(xtest[:, 0], predictions, color='red', alpha=0.5) # Предсказания
ax[1].set_title("Тест vs Предсказание")
plt.show()'''
