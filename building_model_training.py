# Building model and training

from keras.layers import LSTM, Dense, Dropout, Conv2D,BatchNormalization,Reshape, Flatten,concatenate
from keras import Input, Model
from keras.optimizers import Adam
import keras
from sklearn.model_selection import train_test_split
import json
import numpy as np

with open("data.json","r") as json_file:
  data = json.load(json_file)

input_data = np.array(data["MFCCs"])
target = np.array(data["label"])
ind2lab = np.array(data["mapping"])

input_data = input_data.reshape(-1,input_data.shape[1],input_data.shape[2],1)/127.0-1
target = target.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(input_data,target,test_size=0.1, random_state=42)


#Model

input_shape = (input_data.shape[1],input_data.shape[2],1)

input_layer = Input(shape=input_shape)

x = input_layer

filters = [2,4,8,16,32,64,32,16]

for i,filterx in enumerate(filters):
  x = BatchNormalization()(x)
  if i > 5:   
    x = Conv2D(filterx,kernel_size=(3,3),padding="same",strides=(2,2), activation="relu")(x)
  else:
    x = Conv2D(filterx,kernel_size=(3,3),padding="same", activation="relu")(x)

x = Flatten()(x)

x = Dense(units=1000, activation="relu")(x)

x = Dropout(0.3)(x)

x = Dense(units=100, activation="relu")(x)

x = Dropout(0.3)(x)

output_layer = Dense(units=10, activation="softmax")(x)

model = Model(input_layer,output_layer)
model.summary()


from keras.callbacks import ModelCheckpoint
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(0.001),
              metrics=["accuracy"])

filepath = "./model.h5"
save = ModelCheckpoint(
    filepath, verbose=1, save_best_only=True,monitor='val_accuracy')

model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test), verbose=1, callbacks=[save])