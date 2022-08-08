from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


#LeNet architecture
class LeNet:

  @staticmethod
  def build(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)

    #1st block
    model.add(Conv2D(20, (5,5), input_shape = input_shape, padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #2d block
    model.add(Conv2D(50, (5,5), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #3d block
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    return model

#Mini VGG architecture
class MiniVGGNet():
    
  @staticmethod
  def build(height, width, depth, classes):
    model = Sequential()
    inputt = (height, width, depth)
    #1st block
    model.add(Conv2D(32, (3,3), input_shape=inputt, activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    #2d block
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    #3d block
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    return model

  def build_wBN(height, width, depth, classes):
    model = Sequential()
    inputt = (height, width, depth)
    #1st block
    model.add(Conv2D(32, (3,3), input_shape=inputt, activation='relu', padding='same'))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    #2d block
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    #3d block
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    return model
