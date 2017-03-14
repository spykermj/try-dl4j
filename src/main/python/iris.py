from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

datasetx = np.loadtxt("iris.csv", delimiter=',', usecols=(0,1,2,3))
datasety = np.loadtxt("iris.csv", dtype=np.dtype(str), delimiter=',', usecols=(4,))

X = datasetx[:,0:4].astype(float)
Y = datasety

#encode class values as integers
uniques, ids = np.unique(Y, return_inverse=True)
encoded_Y = np_utils.to_categorical(ids, len(uniques))

# convert integers to dummy variables (hot encoded)
dummy_y = encoded_Y

# define baseline model
#def baseline_model():
# create model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(3,activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#    return model
#model.fit
model.fit(X, dummy_y, nb_epoch=200, batch_size=5, verbose=0)

with open('/tmp/iris_model.json', 'w') as o:
    o.write(model.to_json())

model.save_weights('/tmp/iris_weights.h5')

loss, accuracy = model.evaluate(X, dummy_y, verbose=0)
print(loss, accuracy)
