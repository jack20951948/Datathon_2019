import pandas as pd
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras import optimizers

raw_data = pd.read_csv("Datathon2019/Datathon_dataset/mean_train_final.csv")



X = raw_data[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE', 'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num']]
y = np.ravel(raw_data[['die_in_h']])
print(X)

# normalize the data
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)


#DNN
batch_size = 10
nb_epoch = 500
model = Sequential()
model.add(Dense(32, input_dim=29, activation='relu',kernel_regularizer=l2(0.02)))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu',kernel_regularizer=l2(0.02)))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(0.02)))


sgd = optimizers.Adadelta(lr=0.33, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


history = model.fit(x=X, y=y, epochs=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.1)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
model.save('Datathon2019/Datathon_Model/DNN_model_1.h5')
# history_dict = history.history
# print(history_dict.keys())
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()