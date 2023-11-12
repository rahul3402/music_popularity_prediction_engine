import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os 

# Client Keys
CLIENT_ID = "b9da89f4876f4f65acb8e10cc7cacb4b"
CLIENT_SECRET = "18ba2836e0cd4aa189624a46b9deafbd"

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

normal_pop_id = "4n6pJZnZJUGtAzvdG9fvfC"
normal_pop = "https://open.spotify.com/playlist/4n6pJZnZJUGtAzvdG9fvfC?si=6447e9f6b3e346f0"
lowkey_pop_id = "3pxwKjnmDg4kwTc5nItYcx"
lowkey_pop = "https://open.spotify.com/playlist/3pxwKjnmDg4kwTc5nItYcx?si=29147a5985b44665"

def get_playlist_tracks(username,playlist_id):
    results = sp.user_playlist_tracks(username, playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

parth_id = "iz49c589n9qyia3dmpq1n9b2s"
all_tracks = get_playlist_tracks(parth_id, lowkey_pop_id)

for i in all_tracks:
    artist = i["track"]['artists'][0]['name']
    song = i["track"]['name']
    artist = "{} - {}".format(artist, song)
    file = open('lksongs_name.txt','a')
    file.writelines(artist + "\n")

    uri = i['track']['uri']
    if uri.find("track") == -1:
        continue
    portion = uri.partition("track:")[2]
    url = "https://open.spotify.com/track/" + portion
    os.chdir("/Users/rahul/classes/cs4701/get_tracks/lk_songs")
    os.system("spotdl " + url)
    os.chdir("/Users/rahul/classes/cs4701/get_tracks")