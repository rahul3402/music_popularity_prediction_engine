import csv 
import random

#set lengths
lk_len = 343
n_len = 1275 

lk_indices = [i for i in range(lk_len)]
random.shuffle(lk_indices)
lk_test_ind = lk_indices[0::5]
lk_set = set(lk_test_ind)
 
n_indices = [i for i in range(n_len)]
random.shuffle(n_indices)
n_test_ind = n_indices[0::5]
n_set = set(n_test_ind)

# Creating 80 - 10 - 10 traing split

with open('n_fulldata.csv', mode = 'r') as file:
    with open("n_train.csv", mode = "w") as f1:
        with open("n_test.csv", mode = "w") as f2:
            csvFile = csv.reader(file)
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            for row, line in enumerate(csvFile):
                if row in n_set:
                    writer2.writerow(line)
                else:
                    writer1.writerow(line)

with open('lk_fulldata.csv', mode = 'r') as file:
    with open("lk_train.csv", mode = "w") as f1:
        with open("lk_test.csv", mode = "w") as f2:
            csvFile = csv.reader(file)
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            for row, line in enumerate(csvFile):
                if row in lk_set:
                    writer2.writerow(line)
                else:
                    writer1.writerow(line)

with open('n_test_old.csv', mode = 'r') as file:
    with open("n_validation.csv", mode = "w") as f1:
        with open("n_test_new.csv", mode = "w") as f2:
            csvFile = csv.reader(file)
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            for line in csvFile:
                if random.random() > .5:
                    writer1.writerow(line)
                else:
                    writer2.writerow(line)

with open('lk_test_old.csv', mode = 'r') as file:
    with open("lk_validation.csv", mode = "w") as f1:
        with open("lk_test.csv", mode = "w") as f2:
            csvFile = csv.reader(file)
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            for line in csvFile:
                if random.random() > .5:
                    writer1.writerow(line)
                else:
                    writer2.writerow(line)