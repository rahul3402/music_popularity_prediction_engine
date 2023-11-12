from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf 
import numpy as np
import csv 
import math
import pydub
import librosa
import json

"""
Converting to categorical data- helper functions. 
"""
def map_values_5(arr):
    arr = np.where(arr < 100000, 0, arr)
    arr = np.where((arr >= 100000) & (arr < 1000000), 1, arr)
    arr = np.where((arr >= 1000000) & (arr < 10000000), 2, arr)
    arr = np.where((arr >= 10000000) & (arr < 100000000), 3, arr)
    arr = np.where(arr >= 100000000, 4, arr)
    return arr

def map_values_3(arr):
    arr = np.where(arr < 1000000, 0, arr)
    arr = np.where((arr >= 1000000) & (arr < 15000000), 1, arr)
    arr = np.where((arr >= 15000000), 2, arr)
    return arr

#Load up the data
loaded_data_train = np.load('total_train_np.npz')
loaded_data_validation = np.load('total_validation_np.npz')
loaded_data_test = np.load('total_test_np.npz')

total_train_label = loaded_data_train['labels']
total_train_data = loaded_data_train['data']
total_test_label = loaded_data_test['labels']
total_test_data = loaded_data_test['data']
total_validation_label = loaded_data_validation['labels']
total_validation_data = loaded_data_validation['data']

#Convert data to 5 categories
total_train_label_c = map_values_5(total_train_label)
total_validation_label_c = map_values_5(total_validation_label)
total_test_label_c = map_values_5(total_test_label)
total_train_label_c = to_categorical(total_train_label_c, num_classes=5)
total_validation_label_c = to_categorical(total_validation_label_c, num_classes=5)
total_test_label_c = to_categorical(total_test_label_c, num_classes=5)






#############################################################CLASSIFICATION MODELS########################################################
def classification_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())

    s_cnn.add(Dense(5, activation='softmax'))
    return s_cnn 

def class_highfilters_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())

    s_cnn.add(Dense(5, activation='softmax'))
    return s_cnn 

def class_highfilters_dense_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())

    s_cnn.add(Dense(128, activation='relu'))

    s_cnn.add(Dense(5, activation='softmax'))
    return s_cnn 

def class_highfilters_dense_2layers_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())

    s_cnn.add(Dense(128, activation='relu'))
    s_cnn.add(Dense(5, activation='softmax'))
    return s_cnn 


def class_highfilters_dense_3layers_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    s_cnn.add(MaxPooling2D(padding="same"))


    s_cnn.add(Flatten())

    s_cnn.add(Dense(128, activation='relu'))

    s_cnn.add(Dense(64, activation='relu'))

    s_cnn.add(Dense(5, activation='softmax'))
    return s_cnn 



#########################################################MODEL GENERATION FOR CLASSIFICATION ###########################################
#THIS SECTION IS FOR CALIBRATING CLASSIFCATION MODELS. COMMENT IF
#YOU WANT TO TEST A SINGULAR SONG
create_class_graph = True 
model_c = class_highfilters_dense_3layers_cnn()
model_c.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
history_c = model_c.fit(total_train_data, total_train_label_c, epochs=5, 
                    validation_data=(total_validation_data, total_validation_label_c))

test_predictions = model_c.predict(total_test_data)
pred = np.apply_along_axis(np.argmax, axis = 1, arr = test_predictions)
test = np.apply_along_axis(np.argmax, axis = 1, arr = total_test_label_c)
mse_error = np.mean(np.square(pred - test))
# mse_error = np.argmax(test_predictions) - total_test_label_c
print("The mean squared error for categorical outputs is: ", mse_error)

#Output loss values to a csv for graph generation
if create_class_graph:
    epo = len(history_c.history['loss']) + 1
    epochs = [i for i in range(1, epo)]
    loss_list = history_c.history['loss']
    val_loss_list = history_c.history['val_loss']
    accuracy_list = history_c.history['accuracy']
    val_accuracy_list = history_c.history['val_accuracy'] 

    with open('Loss_Class_Graph.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy'])  # Write the headers
        for i in range(epo-1):
            writer.writerow([epochs[i], loss_list[i], val_loss_list[i], accuracy_list[i], val_accuracy_list[i]])

##########################################################################################################################################

#############################################################REGRESSION MODELS########################################################

#1 layer, only 16 filters
def original_cnn():
    s_cnn = Sequential(name="Song_CNN")

    s_cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))

    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())
    s_cnn.add(Dense(1, activation='relu'))
    return s_cnn 

def highfilters_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())
    s_cnn.add(Dense(1, activation='relu'))
    return s_cnn 

def highfilters_dense_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())

    s_cnn.add(Dense(128, activation='relu'))
    s_cnn.add(Dense(1, activation='relu'))
    return s_cnn 

def highfilters_dense_2layers_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Flatten())

    s_cnn.add(Dense(128, activation='relu'))
    s_cnn.add(Dense(1, activation='relu'))
    return s_cnn 

def highfilters_dense_3layers_cnn():
    s_cnn = Sequential(name="Song_CNN")
    s_cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", 
        activation="relu", input_shape=(46, 128, 196)))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    s_cnn.add(MaxPooling2D(padding="same"))

    s_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    s_cnn.add(MaxPooling2D(padding="same"))


    s_cnn.add(Flatten())

    s_cnn.add(Dense(128, activation='relu'))

    s_cnn.add(Dense(64, activation='relu'))
    s_cnn.add(Dense(1, activation='relu'))
    return s_cnn 
#########################################################MODEL GENERATION FOR "REGRESSION" ###############################################
#THIS SECTION IS FOR CALIBRATING REGRESSION MODELS. COMMENT IF
#YOU WANT TO TEST A SINGULAR SONG
#Bool to create graph or not 
create_reg_graph = True
model = highfilters_dense_3layers_cnn()
log_err = "mean_squared_logarithmic_error"
abs_perc_err = "mean_absolute_percentage_error"
model.compile(loss=log_err,
              optimizer='adam')
history = model.fit(total_train_data, total_train_label, epochs=5, 
                    validation_data=(total_validation_data, total_validation_label))
test_predictions = model.predict(total_test_data)
absolute_error = np.mean(np.abs(test_predictions - total_test_label))
print("The absolute mean error is: ", absolute_error)
if create_reg_graph:
    #Output loss values to a csv for graph generation
    epo = len(history.history['loss']) + 1
    epochs = [i for i in range(1, epo)]
    loss_list = history.history['loss']
    val_loss_list = history.history['val_loss']

    with open('Loss_Views_Graph.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'loss', 'val_loss'])  # Write the headers
        for i in range(epo - 1):
            writer.writerow([epochs[i], loss_list[i], val_loss_list[i]])

##################################################################################################################################


#Spectrogram generation functions
def cut_song(song):
  start = 0
  end = len(song)
  song_pieces = []
  while start + 100000 < end:
    song_pieces.append(song[start:start+100000])
    start += 100000
  return song_pieces

def prepare_song(song_path):
  list_matrices = []
  dur = pydub.utils.mediainfo(song_path)["duration"]
  y, sr = librosa.load(song_path,  duration = math.floor(float(dur)))
  song_pieces = cut_song(y)
  for song_piece in song_pieces:
    melspect = librosa.feature.melspectrogram(y = song_piece)
    list_matrices.append(melspect)
  return list_matrices

if __name__ == '__main__':
    
    #Load up the data
    loaded_data_train = np.load('total_train_np.npz')
    loaded_data_validation = np.load('total_validation_np.npz')
    loaded_data_test = np.load('total_test_np.npz')

    total_train_label = loaded_data_train['labels']
    total_train_data = loaded_data_train['data']
    total_test_label = loaded_data_test['labels']
    total_test_data = loaded_data_test['data']
    total_validation_label = loaded_data_validation['labels']
    total_validation_data = loaded_data_validation['data']

    #Convert data to 5 categories
    total_train_label_c = map_values_5(total_train_label)
    total_validation_label_c = map_values_5(total_validation_label)
    total_test_label_c = map_values_5(total_test_label)
    total_train_label_c = to_categorical(total_train_label_c, num_classes=5)
    total_validation_label_c = to_categorical(total_validation_label_c, num_classes=5)
    total_test_label_c = to_categorical(total_test_label_c, num_classes=5)


    ##get the spectrogram data from the mp3
    ##ToDo for rahul 
    demo_song = np.array(prepare_song("Ariana Grande, The Weeknd - off the table.mp3"))
    demo_song.resize(1, 46, 128, 196)


    #Want to create a csv with class accuracy?
    create_class_graph = True 
    model_c = class_highfilters_dense_3layers_cnn()
    model_c.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics='accuracy')
    history_c = model_c.fit(total_train_data, total_train_label_c, epochs=3, 
                        validation_data=(total_validation_data, total_validation_label_c))

    test_predictions = model_c.predict(demo_song)
    pred = np.argmax(test_predictions)
    actual = 3
    #pred = np.apply_along_axis(np.argmax, axis = 1, arr = test_predictions)
    #test = np.apply_along_axis(np.argmax, axis = 1, arr = total_test_label_c)
    mse_error = np.mean(np.square(pred - actual))
    # mse_error = np.argmax(test_predictions) - total_test_label_c
    print("The actual popularity of your song (on a scale of 0 to 4) is: ", actual)
    print("The popularity of your song (on a scale of 0 to 4) is predicted to be: ", pred)
    print("The mean squared error for your prediction is: ", mse_error)

    #Output loss values to a csv for graph generation
    if create_class_graph:
        epo = len(history_c.history['loss']) + 1
        epochs = [i for i in range(1, epo)]
        loss_list = history_c.history['loss']
        val_loss_list = history_c.history['val_loss']
        accuracy_list = history_c.history['accuracy']
        val_accuracy_list = history_c.history['val_accuracy'] 

        with open('Loss_Class_Graph_Demo.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy'])  # Write the headers
            for i in range(epo-1):
                writer.writerow([epochs[i], loss_list[i], val_loss_list[i], accuracy_list[i], val_accuracy_list[i]])


    #################REGRESSION DEMO###########################################################################################
    create_reg_graph = True
    model = highfilters_dense_3layers_cnn()
    log_err = "mean_squared_logarithmic_error"
    abs_perc_err = "mean_absolute_percentage_error"
    model.compile(loss=log_err,
                optimizer='adam')
    history = model.fit(total_train_data, total_train_label, epochs=3, 
                        validation_data=(total_validation_data, total_validation_label))
    test_predictions = model.predict(demo_song)
    total_test_label = 23418985
    absolute_error = np.mean(np.abs(test_predictions - total_test_label))
    print("The actual popularity of the song (in terms of views) is: ", total_test_label)
    print("The popularity of your song (in terms of views) is predicted to be: ", int(test_predictions[0]))
    print("The absolute mean error is: ", absolute_error)
    if create_reg_graph:
        #Output loss values to a csv for graph generation
        epo = len(history.history['loss']) + 1
        epochs = [i for i in range(1, epo)]
        loss_list = history.history['loss']
        val_loss_list = history.history['val_loss']

        with open('Loss_Views_Graph_Demo.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'loss', 'val_loss'])  # Write the headers
            for i in range(epo - 1):
                writer.writerow([epochs[i], loss_list[i], val_loss_list[i]])

