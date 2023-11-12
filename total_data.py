import numpy as np

loaded_data1 = np.load('n_train_np.npz')
loaded_data2= np.load('n_validation_np.npz')
loaded_data3 = np.load('n_test_np.npz')
loaded_data4 = np.load('lk_train_np.npz')
loaded_data5 = np.load('lk_validation_np.npz')
loaded_data6 = np.load('lk_test_np.npz')

data1 = list(loaded_data1['data'])
labels1 = list(loaded_data1['labels'])

data2 = list(loaded_data2['data'])
labels2 = list(loaded_data2['labels'])

data3 = list(loaded_data3['data'])
labels3 = list(loaded_data3['labels'])

data4 = list(loaded_data4['data'])
labels4 = list(loaded_data4['labels'])

data5 = list(loaded_data5['data'])
labels5 = list(loaded_data5['labels'])

data6 = list(loaded_data6['data'])
labels6 = list(loaded_data6['labels'])

total_train_data = np.array(data1 + data4) 
print(total_train_data.shape)
total_validation_data = np.array(data2 + data5)
print(total_validation_data.shape)
total_test_data = np.array(data3 + data6)
print(total_test_data.shape)

total_train_label = np.array(labels1 + labels4)
print(total_train_label.shape)
total_validation_label = np.array(labels2 + labels5)
print(total_validation_label.shape)
total_test_label = np.array(labels3 + labels6)
print(total_test_label.shape)

np.savez('total_train_np.npz', data=total_train_data, labels=total_train_label)
np.savez('total_validation_np.npz', data=total_validation_data, labels=total_validation_label)
np.savez('total_test_np.npz', data=total_test_data, labels=total_test_label)

