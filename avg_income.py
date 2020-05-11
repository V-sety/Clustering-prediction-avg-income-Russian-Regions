import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel('values+avg_income.xlsx')
#df.interpolate(axis = 1, inplace = True)
df.fillna(method = 'bfill', inplace = True)
X = df.drop(['avg income'],1)
y = df.drop(['web-site', 'fixed internet', 'servesies/person', 'investments', 'spending on innovation', 'sent info', 'mobile internet', 'total spends', 'innovations in organizations'],1)

#X = np.array(df.astype(float))
X = StandardScaler().fit_transform(X)
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#print(X_train)
y_train = y_train / 24608.32512

print(y_train)
model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=(9, )),  
    keras.layers.Dense(1) # output layer (3)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss='mse',
            metrics=['mae', 'mse'])

model.fit(X_train, y_train, epochs=100, batch_size=10)

prediction = model.predict(X_test)
prediction = prediction * 24608.32512

print(prediction)
print(y_test)