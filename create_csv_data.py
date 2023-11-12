import csv 
import getdata


lk_id = "lk_ids.txt"
lk_song_names = "lksongs_name_u.txt"

n_id = "n_ids.txt"
n_song_names = "nsongs_name.txt"



if __name__ == '__main__':
    youtube = getdata.youtube_authenticate()

    views_lk = []
    views_n = [] 
    #First handle lowkey songs
    with open(lk_id, 'r') as file:
        ids = file.read().splitlines()
        ids.reverse()
    with open(lk_song_names, 'r') as file:
        names = file.read().splitlines()

    #now output csv
    with open('lk_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Song File Name", "Youtube_ID", "Youtube_Views"])

        for i in range(len(ids)):
            tmp_id = ids[i]
            tmp_name = names[i]
            response = getdata.get_video_details(youtube, id=tmp_id)
            tmp_views = getdata.get_views(response)
            writer.writerow(["432432" + tmp_name + "432433", tmp_id, tmp_views])
    #Now lets handle normal songs
    
    with open(n_id, 'r', encoding='utf-8') as file:
        ids_1 = file.read().splitlines()
        #have to reverse just because of the way it was set up
        ids_1.reverse()
    with open(n_song_names, 'r', encoding='utf-8') as file:
        names_1 = file.read().splitlines()

    with open('n_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Song File Name", "Youtube_ID", "Youtube_Views"])
        for i in range(len(ids_1)):
            tmp_id = ids_1[i]
            tmp_name = names_1[i]
            response = getdata.get_video_details(youtube, id=tmp_id)
            tmp_views = getdata.get_views(response)
            writer.writerow(["432432" + tmp_name + "432433", tmp_id, tmp_views])