from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np
import datetime as dt

from functions import get_features_and_targets
from models import conv_large, conv_small, conv_tiny

features, targets, images = get_features_and_targets('./pics/','data.csv',data_folder='./data')

targets_categorical = np.concatenate([targets[:,None]==0,targets[:,None]==1,targets[:,None]==2],axis=1)

model = conv_tiny.build()

n_test = 2

split = 361*(len(images)-n_test)

perm = np.arange(split)
np.random.shuffle(perm)

X_train,X_test = features[:split][perm], features[split:]
y_train,y_test = targets_categorical[:split][perm],targets_categorical[split:]
X_train,X_test = X_train/255., X_test/255.

class_weight = {0:0.01,1:1,2:0.4}

tensorboard = TensorBoard(log_dir='./Graph/%s'%dt.datetime.now().strftime('%Y%m%d-%H:%M:%s'), histogram_freq=0,  
          write_graph=True, write_images=True)

batch_size=256
samples_per_epoch=4096
model.fit(X_train, y_train, batch_size=batch_size,
                    nb_epoch=300,
                    validation_data=(X_test,y_test),
                    class_weight=class_weight,
                    callbacks=[
                        ModelCheckpoint('saved_models/conv_tiny_best_1.h5',monitor='val_loss',save_best_only=True,verbose=True),
                        tensorboard])
