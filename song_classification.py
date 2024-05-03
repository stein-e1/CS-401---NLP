# -*- coding: utf-8 -*-
"""
11.07.2023
Author / Copyright: Elijah Stein, All Rights Reserved

This program utilizes Naive Bayes classification on the Million Song Dataset
(http://millionsongdataset.com/) to classify songs that are either metal or rock.
Additionaly, we unleash the trained model on a small selection of chatgpt
generated metal and rock songs with different themes. The details of the query
can be seen in the READ_ME.txt file.
"""
import time
import sys
import sqlite3 as sql
import nltk

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

# GLOBALS:
START_TIME = time.time()
TRACK_ID_COLUMN = 2
GENRE_COLUMN = 3
LYRIC_COLUMN = 7

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def classify( data : list, classifier ):
    """
    classifies data and shows accuracy using data and a trained nltk naive bayes
    classifier

    Args:
        data (list): formatted data for nltk naive bayes classification.
        classifier (nltk classifier): trained nltk naive bayes classifier.

    Returns:
        None. Prints classification accuracy.
    """
    accuracy = nltk.classify.accuracy( classifier, data )
    # show acuracy
    print( "Naive Bayes Classifier Accuracy: " + str( accuracy ) + "\n" )

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def create_vectors_and_genres( data_cursor, word_to_vector_position : dict ):
    """
    Builds 2 dictionaries mapping track_id : vector of terms, and track_id :
    genre in one pass over the data.

    Args:
        * data_cursor (sqlite3 cursor): cursor containing table created in main
        which contains artist ids, track ids, genres, individual lyrics, and
        their counts

    Returns:
        * 2 dictionaries mapping track_id : vector of words, and track_id :
        genre. Prints number of tracks used.

    """
    # init our 3 return dictionaries
    track_id_to_vector = {}
    track_id_to_genre = {}

    # get stop words so we can ignore them in our vectors
    stop_words = set( nltk.corpus.stopwords.words() )

    # iterate over each row of data - contains one word and its count per row
    for row in data_cursor:
        # find our current word/lyric -- we will add 1 to the approptiate
        # position in the vector, add the count to our count vector, and the
        # update the genre
        word = row[ LYRIC_COLUMN ]

        # skip stop words:
        if word in stop_words:
            continue

        # find the track id and count -- this is our key in track_id_to_vector
        # and track_id_to_count, respectivley
        track_id = row[ TRACK_ID_COLUMN ]

        # case where no vector exists currently:
        if track_id not in track_id_to_vector:
            # init 2 vectors for our vector based dictionaries:
            vector_a = [ 0 ] * len( word_to_vector_position.keys() )
            # lookup genre:
            genre = row[ GENRE_COLUMN ]
            # add a 1 for our word and genre for our genre:
            vector_a[ word_to_vector_position[ word ] ] = 1
            track_id_to_vector[ track_id ] = vector_a
            track_id_to_genre[ track_id ] = genre

        # case where vector/genre already exists:
        else:
            vector_a = track_id_to_vector[ track_id ]
            # update our vector
            vector_a[ word_to_vector_position[ word ] ] = 1
            # update dict with changed vector
            track_id_to_vector[ track_id ] = vector_a

            # some songs have multiple genres attached -- for example some
            # Metallica songs have an entry for both rock and metal genres. This
            # checks that if a new genre has been detected for a given track id,
            # we can set the genre to False, and use this bool to skip the song
            # in later calculations:
            if track_id_to_genre[ track_id ] != genre:
                track_id_to_genre[ track_id ] = False

    print( "number of tracks = " + str( len( track_id_to_genre.keys() ) ) )
    return track_id_to_vector, track_id_to_genre

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def format_for_naive_bayes( track_id_to_vector : dict, word_to_vector_position : dict, track_id_to_genre : dict ):
    """
    Formats a vector of 1's and 0's representing words and their presence in a
    track to a data format appropriate for nltk classification.

    Args:
        * track_id_to_vector (dict): track ids mapped to vector of 1's and 0's
        * word_to_vector_position (dict): mapping of word to their position in
        the above vectors
        * track_id_to_genre (dict): mapping of track ids to their genre.

    Returns:
        formatted (list): a list of tuples, each representing one track. Each
        tuple containins a dictionary mapping tokens to bools that represent if
        they are present in the track, and a classification of either rock or
        metal. Used for Naive Bayes classification in nltk package formatting.
    """
    # init return list
    formatted = []

    for tid in track_id_to_vector:
        # initialize our tuple elements, each representing one track
        words_to_bool = {}
        genre = track_id_to_genre[ tid ]

        # genre is False when  we have a track with both metal and rock genre
        # tags, we will skip these songs and only look at those with a single tag
        if genre is False:
            continue

        # convert our vector of 1's and 0's to a dictionary mapping words to
        # bools:
        for word in word_to_vector_position:
            position = word_to_vector_position[ word ]

            # if the given word position in our vector is 1, set output dict
            # to true for given word:
            if track_id_to_vector[ tid ][ position ] == 1:
                words_to_bool[ word ] = True
            # ... and otherwise when we see a 0 set to false:
            else:
                words_to_bool[ word ] = False

        # builds and appends tuple:
        tup = ( words_to_bool, genre )
        formatted.append( tup )

    return formatted

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def get_vector_positions( data_cursor ):
    """
    Builds a dictionary mapping each word to a position. These positions are
    used to determine which word is represented in a vector with |number of non
    stop words in input dataest| dimensions.

    Args:
        * data_cursor (sqlite3 cursor): cursor containing table created in main
        which contains artist ids, track ids, genres, individual lyrics, and
        their counts.

    Returns:
        * word_to_vector_position (dict): a mapping of words to positions.

    """
    # used to check for stop words and avoid adding unnecessary dimensions to
    # our vector:
    nltk_stopwords = set( nltk.corpus.stopwords.words() )
    # blank return dictionary
    word_to_vector_position = {}

    count = 0
    for row in data_cursor:
        token = row[ LYRIC_COLUMN ]
        if token not in nltk_stopwords:
            # if the token is a non-stop-word that is not a key yet, then make
            # a key with value count, and iterate count by one.
            if token not in word_to_vector_position.keys():
                word_to_vector_position[ token ] = count
                count += 1

    return word_to_vector_position

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def naive_bayes( track_id_to_vector : dict, word_to_vector_position : dict, track_id_to_genre : dict ):
    """
    Performs naive bayes classification on tracks and their lyrics, classifying
    as either metal or rock genres. Shows most informative features, and accuracy
    of the model on a test data subset.

    Args:
        * track_id_to_vector (dict): track ids mapped to vector of 1's and 0's
        * word_to_vector_position (dict): mapping of word to their position in
        the above vectors
        * track_id_to_genre (dict): mapping of track ids to their genre.

    Returns:
        * None. Prints the accuracy ratio of the model (0-1) where 1 is perfect
        accuracy
    """
    # get the data in readable format for nltk classification functions
    formatted = format_for_naive_bayes(track_id_to_vector, word_to_vector_position, track_id_to_genre)

    # split data into training and testing sets, 20% for training, 80 for test
    length = len( formatted )
    train_threshold = int( length * 0.2 )
    train_data = formatted[0:train_threshold]
    test_data = formatted[train_threshold:]

    # perform training, see accuracy of classification on test set
    classifier = nltk.NaiveBayesClassifier.train( train_data )
    classifier.show_most_informative_features()

    # show accuracy on test_data:
    print( "\ntest data classification:" )
    classify( test_data, classifier )

    # return classifier for use on chatgpt data:
    return classifier

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def stem_and_format( songs : str, genre : str, model_bag_of_words : set ):
    """
    stems words in a string representing several songs, formats them for naive
    bayes classification (list of tuples, one item as a dictionary mapping word
    to a bool representing if its in the song, and a second item as a string
    representing genre -- either metal or rock). Only looks at tokens which are
    in the set of tokens used to train our model.

    Args:
        * songs (str): a formatted string representing some songs seperated by
        a new line character.
        * genre (str): the genre of each song in songs.
        * model_bag_of_words (set): bag of words from used to train a classifer.

    Returns:
        * formatted_list (TYPE): list of each song's bag of word formatted for
        a nltk naive bayes classifier.

    """
    stop_words = set( nltk.corpus.stopwords.words() )
    stemmer = nltk.stem.PorterStemmer()

    # find all non-stop-words in the chatgpt data:
    bag_of_words = set()
    for line in songs:
        for token in line:
            if ( token not in stop_words ) and ( token != "\n" ):
                bag_of_words.add( stemmer.stem( token , True ) )

    # select sub-set which is words in both chatgpt data and our MSD data the
    # naive bayes classifier is trained on:
    shared_words = bag_of_words.intersection( model_bag_of_words )
    print("shared words in " + genre + ": " + str(len(shared_words)))

    # read data and format for naive bayes based on my file formatting:
    formatted_list = []
    lyric_to_bool = {}
    for line in songs:
        # case where we reach a new song; need to append to list and reset dict
        if line == "\n":
            formatted_list.append( ( lyric_to_bool, genre ) )
            lyric_to_bool = {}
        # otherwise, we looked at tokens, make sure they're a valid word, set T
        for token in line:
            if token in shared_words:
                lyric_to_bool[ token ] = True

    # set all lyrics in shared_words which are not in a given song to False:
    for item in shared_words:
        for i in range( len( formatted_list ) ):
            # item is not key in the dictionary of tokens to bools at the ith pos:
            if item not in formatted_list[i][0]:
                formatted_list[i][0][ item ] = False

    return formatted_list

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def main( limit = None ):
    """
    Driver code to generate a useful table from input databases, perform data
    extraction from that table, and then classifies data using naive bayes.
    Applies naive bayes model to several chatgpt generated rock and metal songs.

    Args:
        limit (str): cmd line specified value for how many records to include in
        SQL query.A record contains one token and its count for a given track.

    Returns:
        None.
    """
    # names of databases so we can connect:
    artist_tag_path= "artist_term.db"
    lyrics_path = "mxm_dataset.db"
    metadata_path = "track_metadata.db"
    chatgpt_metal_path = "./chatgpt_metal.txt"
    chatgpt_rock_path = "./chatgpt_alt_rock.txt"

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
    join_sql = """
    WITH join_1 AS (
        SELECT songs.artist_id, songs.artist_name, songs.track_id, artist_tag.artist_mbtag.mbtag, songs.year
        FROM artist_tag.artist_mbtag
        INNER JOIN songs ON artist_tag.artist_mbtag.artist_id = songs.artist_id
        WHERE songs.year != 0 AND (
            artist_tag.artist_mbtag.mbtag LIKE 'metal'
            OR artist_tag.artist_mbtag.mbtag LIKE 'rock'
            )
    )
    SELECT *
    FROM join_1
    INNER JOIN lyrics.lyrics ON lyrics.lyrics.track_id = join_1.track_id
    ORDER BY track_id
    """

    if limit is not None:
        join_sql += "LIMIT " + str( limit )

    # we create 2 cursors because you cannot iterate over a cursor twice, but
    # iterating over a cursor is faster than iterating over a list such as using
    # the cursor.fetchall() method, and running our querey is pretty fast
    metal_cursor = md_cursor.execute( join_sql )
    metal_cursor_2 = md_connection.cursor()
    metal_cursor_2 = metal_cursor_2.execute( join_sql )
    md_connection.commit()

    # generate reference for positions in our vectors which will represent the
    # presence of a given word in a given track:
    word_to_vector_position = get_vector_positions( metal_cursor )
    # build vectors for relevant calculations:
    track_id_to_vector, track_id_to_genre = create_vectors_and_genres( metal_cursor_2, word_to_vector_position )
    # perform naive bayes to see how well we can classify the data:
    classifier = naive_bayes(track_id_to_vector, word_to_vector_position, track_id_to_genre )

    # read chatgpt files:
    with open( chatgpt_metal_path, 'r' ) as file_0:
        chatgpt_metal = file_0.read()
    with open( chatgpt_rock_path, 'r' ) as file_1:
        chatgpt_rock = file_1.read()

    # get tokens used in trained model:
    model_bag_of_words = set( word_to_vector_position.keys() )
    # get formatted data from stemmed chatgpt songs:
    songs_1 = stem_and_format( chatgpt_metal, 'metal', model_bag_of_words )
    songs_2 = stem_and_format( chatgpt_rock, 'rock', model_bag_of_words )
    all_songs = songs_1 + songs_2
    # classify chatgpt data:
    print( "\nchatgpt classification:" )
    classify( all_songs, classifier )

    # close connections:
    md_cursor.close()
    metal_cursor.close()
    metal_cursor_2.close()
    md_connection.close()

    # display time to complete:
    print( "time to complete: " + str( time.time() - START_TIME ) + " sec." )

# take optional cmd line argument and run:
try:
    record_limit = sys.argv[1]
    main( record_limit )
except:
    main()
