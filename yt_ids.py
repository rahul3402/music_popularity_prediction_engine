import csv

def find_all(a_string, sub):
    result = []
    k = 0
    while k < len(a_string):
        k = a_string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1 
    return result

with open('nsongs_youtube.txt','r') as file:
    ids = file.read()
    res = find_all(ids, "watch?v=")
    file = open('lk_ids.txt','a')

    for val in res:
        file.writelines(ids[val + 8: val + 19] + "\n") 