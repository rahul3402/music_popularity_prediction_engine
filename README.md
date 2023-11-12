Thank you for taking the time to explore our project! The final report is included
in the zip as a pdf. Please read the following steps in order to use our project:

Step 1. Make sure you have all the necessary libraries installed in song_cnn.py

Step 2. Ensure you have all the necessary binary files for the train, test, and 
validation data. These will automatically be included in this zip. 

Step 3. Choose the models you would like to use along with the number of epochs
you would like to train for. Also, utilize lines 136 and 244 to determine if you want
to output CSV's of the training loss and validation loss. You can use these to generate graphs to
use as a visual aid.  We recommend the most complex models
we built (these are named appropriately so they are easy to find, and the code
comes defaulted to these models). 

Step 4: If you want to test only a single datapoint, make sure you comment out 
the "Generation for Classification" and "Generation for Regression" sections. These
are only used to test/tune various models and are unnecessary if you want to test 
only a single song. 

Step 5: Ensure you have a desired mp3 file this same directory. Put the name of the file in Line 316.

Step 6: run the code! simply type "python3 song_cnn.py" or "python song_cnn.py" into
the terminal. The results will be neatly printed for you after the models finish training.
