# Music Popularity Prediction Engine

Trained convolutional neural networks with spectrograms extracted from songs to predict popularity metrics across Youtube and Spotify for new songs


## How does it work?

- Make sure you have all the necessary libraries installed in song_cnn.py

- Ensure you have all the necessary binary files for the train, test, and 
validation data (these aren't in the repo since they are too big).

- Choose the models you would like to use along with the number of epochs
you would like to train for. Also, utilize lines 136 and 244 to determine if you want
to output CSV's of the training loss and validation loss. You can use these to generate graphs to
use as a visual aid.  We recommend the most complex models
we built (these are named appropriately so they are easy to find, and the code
comes defaulted to these models). 

- If you want to test only a single datapoint, make sure you comment out 
the "Generation for Classification" and "Generation for Regression" sections. These
are only used to test/tune various models and are unnecessary if you want to test 
only a single song. 

- Ensure you have a desired mp3 file this same directory. Put the name of the file in Line 316.

- Run the code! simply type "python3 song_cnn.py" or "python song_cnn.py" into
the terminal. The results will be neatly printed for you after the models finish training.
