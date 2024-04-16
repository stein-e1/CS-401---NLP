# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:21:52 2023

@Author/Copyright - Elijah Stein, Dagmawi Zerihun, All rights reserved 
"""
#0123456789012345678901234567890123456789012345678901234567890123456789012345678

# error E0401 is unable to import, which is not relevant because all imports work
# pylint: disable=E0401
# pylint: disable=trailing-whitespace

import time
import sys
import sqlite3 as sql
import numpy as np
import nltk
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

# globals:
START_TIME = time.time()
TRACK_ID_COLUMN = 2
GENRE_COLUMN = 3
LYRIC_COLUMN = 7

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def convert_to_dataframe( dataset : list ):
    """
    Converts a list formatted for nltk's naive bayes to a pandas dataframe
    suitable for our neural network function.

    Args:
        dataset ( list ): our data initially extracted in a format suitable for
        nltk naive bayes classification

    Returns:
        pd.DataDrame: A pandas dataframe where each record is a track,
        each column is the genre (first column) or a lyric, and each cell is the
        boolean value of the lyric's presence
    """
    # set up initial pandas formatting
    data = { 'genre': [ track[ 1 ] for track in dataset ] }

    for index, (word_dict, _) in enumerate( dataset ):
        # get the word and the bool associated with its presence in the bag of
        # words:
        for word, presence in word_dict.items():
            if word not in data:
                data[ word ] = [ False ] * len( dataset )
            data[ word ][ index ] = presence

    return pd.DataFrame( data )

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def neural_net( data_frame, batch_size : int ):
    """
    Builds and runs a neural network using a pandas dataframe in the format
    where each record is a track, each column is the genre (first column) or a 
    lyric, and each cell is the boolean value of the lyric's presence / the name
    of the genre. Runs NN on a 20% testing subset of the data after training on
    80%.

    Args:
        * formatted_data (Pandas DataFrame): pandas dataframe of genres and lyrics
        formatted to the above specifications.
        * batch_size (int): the batch size used to compile the model. Should be
        a power of 2. The greater the value, the more memory intensive the
        computation, but the less time it takes.

    Returns:
        None. Prints the accuracy of generated TensorFlow model on the testing
        data.
    """
    # Extract features (lyrics) and labels (genres)
    # Features:
    features = data_frame.iloc[:, 1:].values
    # Lables:
    genres = pd.factorize( data_frame[ "genre" ] )[ 0 ]
    
    # Split the data into training and testing sets, 80 for training, 20 testing
    # and set seed for reproducibility:
    np.random.seed( 42 )
    mask = np.random.rand( len( data_frame ) ) < 0.8
    # this is some fancy code using the ~ operator to invert boolean values and
    # ensure that all elements are either in the training or testing set, but
    # never both:
    features_train, genres_train = features[ mask ], genres[ mask ]
    features_test, genres_test = features[ ~mask ], genres[ ~mask ]
    
    # Build the RNN model:

    # embedding_dim represents the dimensions returned by the embedding layer,
    # or the level of detail the model will extract from a given feature
    embedding_dim = 50
    model = Sequential()
    # creation specification of the model were inspired by tensorflow
    # documentation examples
    model.add(Embedding(input_dim = 2, 
                        output_dim = embedding_dim, 
                        input_length = features.shape[ 1 ]))
    # LSTM layers process sequences. They maintain a 'memory' of prior inputs in
    # a sequence
    # units = 100 represents the number of neurons in the layer, and was tuned
    # for a balance of efficiency and speed:
    model.add( LSTM ( units = 100 ) )
    # via TensorFlow documentation, Dense layers "do some final processing, and
    # convert from this vector representation to a single logit as the 
    # classification output." The 100 LSTM layer neurons feed into 2 neurons,
    # one per genre.
    model.add(Dense( units = len( data_frame[ "genre" ].unique() ), activation = "softmax" ) )
    
    # Compile the model
    model.compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics = [ "accuracy" ] )
    
    # Use EarlyStopping - stop training when improvments become marginal
    # patience represents the number of epochs to wait after the last time the
    # monitored metric (like validation loss) improved before stopping the
    # training:
    early_stopping = EarlyStopping( monitor= 'val_loss', patience = 3)

    # Train the model on the training set:
    epochs = 10
    model.fit( features_train, genres_train, epochs = epochs, batch_size = batch_size, 
              validation_split = 0.2, callbacks = [ early_stopping ] )
    
    # Evaluate the model on the testing set:
    predictions = model.predict( features_test )
    accuracy = np.mean( np.argmax( predictions, axis = 1 ) == genres_test )
    
    # Display accuracy:
    print( "RNN Test Accuracy: " + str( accuracy ) )

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def naive_bayes( naive_bayes_data : list ):
    """
    Performs naive bayes classification on tracks and their lyrics, classifying
    by genres. Shows most informative features, and accuracy of the model on a
    test data subset.
    
    Taken and modified from Elijah Stein's "song_classification" program 

    Args:
        naive_bayes_data (list): A list of tuples, each representing a track.
        Element 0 is the bag of words for the track, represented as a dictionary
        of words to bools. Element 1 is the genre tag.

    Returns:
        None. Prints relevant info about the naive bayes classification process,
        including accuracy as a ratio and the most informative classification
        features.
    """
    # split data into training and testing sets, 20% for training, 80 for test
    length = len( naive_bayes_data )
    train_threshold = int( length * 0.2 )
    train_data = naive_bayes_data[ 0 : train_threshold ]
    test_data = naive_bayes_data[ train_threshold: ]

    # perform training, see accuracy of classification on test set
    classifier = nltk.NaiveBayesClassifier.train( train_data )
    classifier.show_most_informative_features()

    # show accuracy on test_data:
    print( "\ntest data classification:" )
    accuracy = nltk.classify.accuracy( classifier, test_data )
    # show acuracy:
    print( "Naive Bayes Classifier Accuracy: " + str( accuracy ) + "\n" )

#0123456789012345678901234567890123456789012345678901234567890123456789012345678    

def read_cursor( data_cursor ):
    """
    Reads data from a subset of the Million Song Datast (MSD) into a format
    suitable for nltk's naive bayes model.

    Args:
        data_cursor (sqlite3 cursor): iterable cursor that goes over each record
        of the MSD table.

    Returns:
        naive_bayes_formatting (list): List of tuples. Each tuple has element 0
        equal to a dictionary mapping unique words in the MSD subset to a bool,
        showing if that word is present in the track. Element 1 is a genre tag.
        Each tuple is representative of a single track.
    """

    naive_bayes_formatting = []
    stop_words = set( nltk.corpus.stopwords.words() )
    non_stop_lyrics = set()
    seen_tracks = {}

    for row in data_cursor:
        
        # get relevant info from the row:
        word = row[ LYRIC_COLUMN ]
        track_id = row[ TRACK_ID_COLUMN ]
        genre = row[ GENRE_COLUMN ]

        # skip stop words:
        if word in stop_words:
            continue

        # keep track of uniqie lyrics:
        non_stop_lyrics.add( word )

        # case where we have already seen this track, set new word to True:
        if track_id in seen_tracks:
            position = seen_tracks[ track_id ]
            naive_bayes_formatting[ position ][ 0 ][ word ] = True

        # case where we are seeing the track for the first time:
        else:
            # update seen_tracks with tid as key and index of this tid as value:
            seen_tracks[ track_id ] = len( naive_bayes_formatting ) - 1
            tup_0 = { word : True }
            tup_1 = genre
            tup = ( tup_0, tup_1 )
            naive_bayes_formatting.append( tup )

    # this code iterates through each track and fills in missing words as False:
    for i in range( len( naive_bayes_formatting ) ):
        word_to_bool = naive_bayes_formatting[ i ][ 0 ]

        for word in non_stop_lyrics:
            # when the word is not in the song...
            if word not in word_to_bool:
                # we need to set the value of this word to False:
                word_to_bool[ word ] = False

        # update dictionary found in the tuple at poition i:
        tup_list = list( naive_bayes_formatting[ i ] )
        tup_list[ 0 ] = word_to_bool
        tup_list = tuple( tup_list )
        naive_bayes_formatting[ i ] = tup_list

    return naive_bayes_formatting

#0123456789012345678901234567890123456789012345678901234567890123456789012345678    

def main( limit : int = None ):
    """
    Driver code to generate a useful table from input databases, perform data
    extraction from that table, and then classifies data using naive bayes and a
    neural network.
    
    Taken and modified from Elijah Stein's "song_classification" program

    Args:
        limit (int, optional): Adds a qualifier to SQL data extraction that
        limits the number of records in the MSD processed. Defaults to None,
        meaning all records are processed. This causes a longer run-time.

    Returns:
        None. Functions called print relevant output describing naive bayes and
        neural net models.

    """
    artist_tag_path= "artist_term.db"
    lyrics_path = "mxm_dataset.db"
    metadata_path = "track_metadata.db"
    
    # connect to a database. We will join other databases to this connection.
    md_connection = sql.connect( "./" + metadata_path )
    md_cursor = md_connection.cursor()

    # attach other databases to our connection
    attach_sql = "ATTACH DATABASE '" + artist_tag_path + "' AS artist_tag"
    attach_sql_2 = "ATTACH DATABASE '" + lyrics_path + "' AS lyrics"
    md_cursor.execute( attach_sql )
    md_connection.commit()
    md_cursor.execute( attach_sql_2 )
    md_connection.commit()

    # build SQL statment to select relevant data from connection:
    # change the genres as needed on different runs to test:
    join_sql = """
    WITH join_1 AS (
        SELECT songs.artist_id, songs.artist_name, songs.track_id, 
            artist_tag.artist_mbtag.mbtag, songs.year
        FROM artist_tag.artist_mbtag
        INNER JOIN songs ON artist_tag.artist_mbtag.artist_id = songs.artist_id
        WHERE songs.year != 0 AND (
            artist_tag.artist_mbtag.mbtag LIKE 'rock'
            OR artist_tag.artist_mbtag.mbtag LIKE 'metal'
        )
    )
    SELECT *
    FROM join_1
    INNER JOIN lyrics.lyrics ON lyrics.lyrics.track_id = join_1.track_id
    ORDER BY track_id
    """
    
    if limit is not None:
        join_sql += " LIMIT " + str( limit )
    
    final_cursor = md_cursor.execute( join_sql )
    formatted_nb = read_cursor( final_cursor )
    
    naive_bayes( formatted_nb )

    data_frame = convert_to_dataframe( formatted_nb )
    neural_net( data_frame, 128 )
    
    md_cursor.close()
    md_connection.close()
    print( str( time.time() - START_TIME ) + " seconds" )

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

try:
    RECORD_LIMIT = sys.argv[1]
    main( RECORD_LIMIT )
except:
    main()
