# -*- coding: utf-8 -*-
"""
10/12/2023
Author/Copyright: Elijah Stein, all rights reserved

This program creates a putative thesaurus based on words in the english language
and a set of keywords that the definitions must contain. The dictionary used is
the NLTK wordnet dictionary.

Note: this program took ~3 minutes to complete a thesaurus with 9 catagories
when checking matches for same part of speech. The machine it ran on was a Dell
laptop from ~2018 with 4 cores and 8 threads. On the same machine, not checking
for equivalent parts of speech, the same program ran in ~1 minute.
"""

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

from nltk.corpus import words, wordnet, stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def get_jaccard( word_a : str, word_b : str, words_to_definitions : dict ):
    """
    Utilizes Jaccard's Coefficient to calculate the measure of similarity
    between two word's definitions as a bag of words.

    Args:
        * word_a (str): the first word
        * word_b (str): the second word
        * words_to_definitions (dict): a dict mapping words to their definition
        bag of words

    Returns:
        * similarity (float): the ratio of similarness between bag of words for
        definitions of word_a and word_b calculated w/ Jaccard's Coefficient'

    """
    # get definition bag of words for both passed words
    word_a_definition = words_to_definitions[ word_a ]
    word_b_definition = words_to_definitions[ word_b ]

    # calculate the intersection of bags of words and number of in common words
    in_common = word_a_definition.intersection( word_b_definition )
    num_in_common = len( in_common )

    # calculate the union of bags of words and the total number of words
    all_elements = word_a_definition.union( word_b_definition )
    num_all_elements = len( all_elements )

    # use the above calculations to get a similarity ratio, checking for
    # division by 0 first
    if num_all_elements != 0:
        similarity = num_in_common / num_all_elements
        return similarity

    # return 0 when the num_all_elements is 0:
    return 0

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def get_wordnet_pos( nltk_pos : str ):
    """
    Converts a nltk pos tag to either "n" - noun, "j" - adjective, "v" - verb,
    or "r" - adverb.

    Args:
        * nltk_pos (str): a string representing a nltk part of speech.

    Returns:
        * str: a corresponding string representing the wordnet part of speech, 
        or None if no pos was found.

    """

    first_letter = nltk_pos[0]
    if first_letter in ( "N", "n" ):
        return "n"
    elif first_letter == ("J", "j" ):
        return "a"
    elif first_letter in ( "V", "v" ):
        return "v"
    elif first_letter in ( "R", "r" ):
        return "r"
    return None

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def get_similarity_dict( words_to_definitions: dict, intersection_tolerance: float, check_pos: bool ):
    """
    Generates and return a dictionary mapping mapping a word : a list of tuples
    containing ( synonym, intersection ratio of definition bag of words ).

    Args:
        * words_to_definitions (dict): mapping of words(str) to non-stop-words 
        in their definition(set).
        * intersection_tolerance(str): the minimum ratio of intersection between
        words in definitions of two words to be considered synonymous.
        * check_pos (bool): decides of synonyms of a word must match the pos of
        the given word. Increases compute time significantly.
    Returns:
        * A dictionary mapping a tuple of 2 words in alphabetic order to the 
        pct of non-stop-words that are the same in their definitions
    """

    keys = words_to_definitions.keys()
    meets_intersection_toletence = {}
    for key in keys:
        # generate part of speech tag if checking for equivalent parts of speech
        if check_pos:
            key_pos = pos_tag( [key] )[0][1]
            key_pos = get_wordnet_pos( key_pos )

        other_keys = keys
        for other_key in other_keys:

            # move to next iteration if we are checking definitions of same
            # words
            if other_key == key:
                continue

            # otherwise, get the similarity and add to our output dict when
            # tolerance criteria is met and parts of speech match (if the user
            # has specified they want to match pos):

            # get pos and check for match, otherwise have a defualt True value
            pos_is_equal = True
            if check_pos:
                other_key_pos = pos_tag( [other_key] )[0][1]
                other_key_pos = get_wordnet_pos( other_key_pos )
                pos_is_equal = key_pos == other_key_pos

            # get similarity
            similarity = get_jaccard( key, other_key, words_to_definitions )

            # when similarity threshold is met and pos matches...
            if ( similarity >= intersection_tolerance and pos_is_equal ):
                # round to 2 decimals for printing:
                similarity = round( similarity, 2 )

                # format tuple that includes a synonym and its Jaccard coeff.
                similarity_tuple = ( other_key, similarity )

                # add similarity_tuple to list that is the value in
                # meets_intersection_tolerance:
                if key not in meets_intersection_toletence:
                    meets_intersection_toletence[ key ] = [ similarity_tuple ]
                else:
                    value = meets_intersection_toletence[ key ]
                    value.append( similarity_tuple )
                    meets_intersection_toletence[ key ] = value

    return meets_intersection_toletence

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def lemmatize( word : str, lemmatizer : WordNetLemmatizer() ):
    """
    Lemmatizes a word using the imported WordNetLemmatizer. Words with parts of
    speech that cannot be identified will be assumed as nouns.

    Args:
        * word (str): a word to be lemmatized
        * lemmatizer (WordNetLemmatizer()): A WordNetLemmatizer object

    Returns:
        * word (str): the lemmatized version of argument word

    """
    #get part of speech (pos):
    word_pos_nltk = pos_tag( [word] )[0][1]
    word_pos_wordnet = get_wordnet_pos( word_pos_nltk )
    # if pos was found, use it to lemmatize, otherwise pass pos as default "n"
    if word_pos_wordnet:
        word = lemmatizer.lemmatize( word, pos = word_pos_wordnet )
    else:
        word = lemmatizer.lemmatize( word )
    return word

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def main( intersection_tolerence : float, catagories: list, check_pos : bool ):
    """
    Generates and prints a thesaurus based upon a list of keywords. Thesaurus
    is in the format of a dictionary mapping a word : a list of tuples
    containing ( synonym, intersection ratio of definition bag of words ).

    Args:
        * intersection_tolerence (float): the minimum ratio of intersection 
        between words in definitions of two words to be considered synonymous.
        * catagories (list): a list of catagories the user wants to be included
        in the definition of the words in their generated thesaurus. For ex. a
        musical thesaurus might take the list 
        ["music", "musical", "chord", "song", "sound"] as argument, and then
        only words with at least one of the elemnets of the list in their def
        would be considered.
        * check_pos (bool): decides of synonyms of a word must match the pos of
        the given word. Increases compute time significantly.

    Returns:
        None. Prints thesaurus as word: list of synonyms and their intersection
        ratio in a tuple

    """
    # get set of non-stop-words using nltk dictionaries and set operations:
    stop_words = set( stopwords.words( "english" ) )
    all_words = set( words.words() )
    non_stop_words = all_words.difference( stop_words )
    # blank dictionary that will store words as keys and set of definition words
    # as values:
    dictionary_of_words = {}
    # build a custom tokenier using regex that excludes punctuation
    tokenizer = RegexpTokenizer( r"\w+" )
    # initalize one lemmatizer object for future recurring computations
    lemmatizer = WordNetLemmatizer()

    for word in non_stop_words:
        # lemmatize words, lower case words for consistency, generate synset
        # object for word that contains a definition:
        word = word.lower()
        word = lemmatize( word, lemmatizer )
        synset = wordnet.synsets( word )

        # ensure a definition exists for the "word" - passes tokens that are
        # actually numbers, hyphenized combonations of words, or anything that
        # won't have a definition - some words might be lost due to aggressive
        # stemming:
        try:
            definition = synset[0].definition()
        except:
            # go to next loop iteration if no definition exists - skips token:
            continue

        # split definition into tokens:
        tokenized_def = tokenizer.tokenize( definition )

        # exclude words without user any defined catagories in definition:
        if any( cat in tokenized_def for cat in catagories ):
            # convert definition tokens to lower case:
            tokenized_def = [ t.lower() for t in tokenized_def ]
            # remove stop-words from tokenization of definition`
            tokenized_def = [ w for w in tokenized_def if w not in stop_words ]

            # lemmatize each token of definition:
            lemmatized_def = []
            for token in tokenized_def:
                lemmatized_token = lemmatize( token, lemmatizer )
                lemmatized_def.append( lemmatized_token )

            # map word to set of lemmatized non-stop-words in defitition:
            dictionary_of_words[ word ] = set( lemmatized_def )

    # generate a dictionary mapping words to a list of tuples containing
    # 0) a word that is synonymous based on intersection_tolerence, and 1)
    # the intersection similarity calculated using Jaccard's coefficient:
    similarity_dict = get_similarity_dict( dictionary_of_words, intersection_tolerence, check_pos )

    # print output (words and their synonyms + intersection coeff.) to console:
    keys = similarity_dict.keys()
    if len( keys ) == 0:
        print( "No synonymous terms found =(" )

    else:
        for key in similarity_dict:
            value = similarity_dict[ key ]
            print( key + ": " + str( value ) )
        print( "Entries in thesaurus: " + str( len( keys ) ) )

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

# tune intersection_tolerance based on human identified error over multiple
# runs of the program
tolerence = 2/3
keywords = ["music",
            "musical",
            "chord",
            "song",
            "sound",
            "rythm",
            "tempo",
            "melody",
            "timbre"
            ]
pos_check = False
main( tolerence, keywords, pos_check )
