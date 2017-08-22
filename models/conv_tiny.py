from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

def build(input_shape=(28,28,3)):

    model = Sequential()

    model.add(Conv2D(8,7,7,activation='relu',input_shape=input_shape,border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(p=0.2))
    model.add(Dense(3,activation='softmax'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model
