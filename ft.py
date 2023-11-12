import pydub
import librosa
import math
import csv
import json
import numpy as np

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

with open('n_data.csv', mode ='r')as file:
  with open('n_fulldata.csv', 'w') as f:
    csvFile = csv.reader(file)
    writer = csv.writer(f)
    for line in csvFile:
      path = "normal_songs/" + line[0]
      ft = prepare_song(path)
      line.append(ft)
      writer.writerow(line)

with open('lk_data.csv', mode ='r')as file:
  with open('lk_fulldata.csv', 'w') as f:
    csvFile = csv.reader(file)
    writer = csv.writer(f)
    for line in csvFile:
      path = "lk_songs/" + line[0]
      ft = prepare_song(path)
      line.append(ft)
      writer.writerow(line)

with open('data/n_data.csv', mode ='r')as file:
  with open('n_fulldata_exp.csv', 'w') as f:
    csvFile = csv.reader(file)
    writer = csv.writer(f)
    for line in csvFile:
      path = "normal_songs/" + line[0]
      ft = prepare_song(path)
      json_arr = json.dumps(ft)
      line.append(json_arr)
      writer.writerow(line)


# n_test
n_test_feature = []
n_test_labels = []

with open('n_test.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for line in csvFile:
      path = "normal_songs/" + line[0]
      ft = np.array(prepare_song(path)) 
      ft.resize(46, 128, 196)
      n_test_feature.append(ft)
      n_test_labels.append(int(line[2]))

n_test_feature = np.array(n_test_feature)
n_test_labels= np.array(n_test_labels)

np.savez('n_test_np.npz', data=n_test_feature, labels=n_test_labels)


# n_train
n_train_feature = []
n_train_labels = []

with open('n_train.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for line in csvFile:
      path = "normal_songs/" + line[0]
      ft = np.array(prepare_song(path)) 
      ft.resize(46, 128, 196)
      n_train_feature.append(ft)
      n_train_labels.append(int(line[2]))

n_train_feature = np.array(n_train_feature)
n_train_labels= np.array(n_train_labels)

np.savez('n_train_np.npz', data=n_train_feature, labels=n_train_labels)

# n_validation
n_validation_feature = []
n_validation_labels = []

with open('n_validation.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for line in csvFile:
      path = "normal_songs/" + line[0]
      ft = np.array(prepare_song(path)) 
      ft.resize(46, 128, 196)
      n_validation_feature.append(ft)
      n_validation_labels.append(int(line[2]))

n_validation_feature = np.array(n_validation_feature)
n_validation_labels= np.array(n_validation_labels)

np.savez('n_validation_np.npz', data=n_validation_feature, labels=n_validation_labels)

#lk_test
lk_test_feature = []
lk_test_labels = []

with open('lk_test.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for line in csvFile:
      path = "lk_songs/" + line[0]
      ft = np.array(prepare_song(path)) 
      ft.resize(46, 128, 196)
      lk_test_feature.append(ft)
      lk_test_labels.append(int(line[2]))

lk_test_feature = np.array(lk_test_feature)
lk_test_labels= np.array(lk_test_labels)

np.savez('lk_test_np.npz', data=lk_test_feature, labels=lk_test_labels)

#lk_train
lk_train_feature = []
lk_train_labels = []

with open('lk_train.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for line in csvFile:
      path = "lk_songs/" + line[0]
      ft = np.array(prepare_song(path)) 
      ft.resize(46, 128, 196)
      lk_train_feature.append(ft)
      lk_train_labels.append(int(line[2]))

lk_train_feature = np.array(lk_train_feature)
lk_train_labels= np.array(lk_train_labels)

np.savez('lk_train_np.npz', data=lk_train_feature, labels=lk_train_labels)

# lk_validation
lk_validation_feature = []
lk_validation_labels = []

with open('lk_validation.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for line in csvFile:
      path = "lk_songs/" + line[0]
      ft = np.array(prepare_song(path)) 
      ft.resize(46, 128, 196)
      lk_validation_feature.append(ft)
      lk_validation_labels.append(int(line[2]))

lk_validation_feature = np.array(lk_validation_feature)
lk_validation_labels= np.array(lk_validation_labels)

np.savez('lk_validation_np.npz', data=lk_validation_feature, labels=lk_validation_labels)