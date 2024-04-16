# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:14:27 2023

This program runs a frequency count on reporting verbs in two texts. Further, 
it analyzes the similarity of the top 20 verbs between both texts, and prints
tables displaying the normalized frequencies and average commonality in English
of these top 20 reporting verbs.

Author / Copyright: Elijah Stein, All Rights Reserved
"""

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

from glob import glob
from collections import Counter

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def find_avg_word_lengths( corpus: list ):
    """
    Takes a lists of words and finds the average word length across all words

    Args:
        * corpus (list): List of pairs with a word as item 0 and freq as item 1

    Returns:
        * average (float): Average word count across all words in corpus

    """
    number_of_words = 0
    total_letters = 0

    # find the total letters and total words in the corpus passed in
    for i in range( len(corpus) ):
        total_letters += len( corpus[i][0] ) * corpus[i][1]
        number_of_words += corpus[i][1]

    # avoid division by 0 error, calculate average word length
    if number_of_words != 0:
        average = total_letters / number_of_words
        return average
    return 0

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def find_same_words( corpus_1 : list, corpus_2: list):
    """
    Identifies words shared between 2 sets of words and their frequencies

    Args:
        * corpus_1 (list): List of pairs with a word as item 0 and freq as 
        item 1
        * corpus_2 (list): List of pairs with a word as item 0 and freq as 
        item 1

    Returns:
        words_in_both_lists (list): List of shared words.

    """
    # grab the words only from our corpus pairs
    just_words_1 = [w[0] for w in corpus_1]
    just_words_2 = [w[0] for w in corpus_2]

    # find the words that are in both lists of words
    words_in_both_lists = []
    for word in just_words_1:
        if word in just_words_2:
            words_in_both_lists.append(word)
    return words_in_both_lists

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def get_reporting_verbs( path_to_reporting_verbs : str ):
    """
    Initializes a counter with keys from a filepath containing a series of 
    reporting verbs, with one such verb per line in the relevent file.

    Args:
        path_to_reporting_verbs (str): A path to the file with reporting verbs

    Returns:
        * reporting_verb_counter (Counter): A mapping of reporting verbs to 
        the number 0

    """
    reporting_verb_counter = Counter()

    # for eac line (containing a reporting verb) intiialize a key in our
    # counter
    with open( path_to_reporting_verbs, encoding="utf-8" ) as file:
        for line in file:
            line = line.strip() #remove newline character
            reporting_verb_counter[line.lower()] = 0
    return reporting_verb_counter

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def get_reporting_verb_count( filenames: list, reporting_verb_path : str ):
    """
    Function to read the text in a series of filenames and count the reporting
    verbs
    Borrows structure from Duncan Buell's' "frequencycountssentpos" program
    Parameters:
        * filenames (list): all the filenames pointing to the corpus of text
        * reporting_verb_path (str): path to reporting verb file with one verb
        per line

    Returns:
        * Counter mapping reporting verbs to their count across the given files
        * Total count of tokens read across files
    """
    tokens_read = 0
    reporting_verbs = get_reporting_verbs( reporting_verb_path )
    for file in filenames:
        with open( file, encoding="utf-8" ) as current_file:
            line_count = 0
            for line in current_file:
                # we ignore even lines which will not contain POS tags,
                # we want to use POS tags to verify a word is actually a verb
                if line_count % 2 != 0:
                    # There are 5 tags (indexed from 0) before the words of
                    # each sentence
                    tokens = line.split()
                    for token in tokens[6:]: # inclusive of index 6 thru end
                        tokens_read += 1
                        word_and_pos = token.split("_")
                        word = word_and_pos[0].lower()
                        pos = word_and_pos[1]

                        # check if a token is a verb and a reporting verb
                        if ( pos[0] == "V" and word in reporting_verbs.keys() ):
                            reporting_verbs[word] += 1
                            #print("reporting verb found")
                line_count += 1

    return reporting_verbs, tokens_read

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def glob_file_names( path_to_data : str, sub_string : str ):
    """ 
    Function to glob filenames from a path to a directory.
    Attribution to Duncan Buell, all rights reserved
    
    Args:
        path_to_data (str): the path to the directory with the files
        sub_string (str): a substring to match in the actual file name
    Returns:
        a 'list' of the filenames
    """
    file_names = glob(path_to_data + '/*' + sub_string + '*')
    return file_names

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def print_formatted_frequencies( reporting_verbs : Counter, total_tokens : int, top_n : int ):
    """
    Prints a table containing the most n common reporting verbs in a counter,
    the count of that verb, and the frequency of that verb per 1000 tokens
    
    Args:
        * reporting_verbs (Counter): Mapping of reporting verbs to their freq
        per 1000 tokens in a corpus
        * total_tokens (int): The total number of tokens in the text containing
        our reporting verbs
        * top_n (int): the top n reporting verbs to display in table

    Returns:
        Nothing, prints formatted string to the console

    """
    # print table header, widths based on number of chars in each column head
    headers = ["Reporting Verbs", "Count", "Frequency / 1000"]
    print("| {:15} | {:5} | {:16} |".format(*headers))

    # iterate over keys and counts, print key as reporting verb and calculate
    # frequency per 1000 from count, print count and frequency as columns
    for key, count in reporting_verbs.most_common( n = top_n ):
        freq_per_thousand = ( count / total_tokens ) * 1000
        pretty_frequency = round( freq_per_thousand, 3 )
        row = [key, count, pretty_frequency]
        print("| {:<15} | {:^5} | {:^16} |".format(*row))

    delineator = "-" * 47
    print(delineator)

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def print_formatted_commonality( counter: Counter ):
    """
    Prints a table containing the descending most common reporting verbs in a 
    counter, as well as how common that verb is in English on a scale from 0-1. 
    Also prints to console the average commonality of all verbs in the counter.

    Args:
        counter (Counter): mapping of reporting verbs to their commonality in
        English (0 - 1).

    Returns:
        None, prints formatted output to console

    """
    # print table header, widths based on number of chars in each column head
    headers = ["Reporting Verbs","Commonality"]
    print("| {:15} | {:11} |".format(*headers))

    # iterate over keys and counts, print key as reporting verb and count as
    # commonality with same formating as header
    for key, commonality in counter.most_common():
        pretty_commonality = round( commonality, 5 )
        row = [key, pretty_commonality]
        print("| {:<15} | {:^11} |".format(*row))

    # finally, calculate and print the average value of the commonality across
    # all keys in counter
    average_commonality = sum(counter.values()) / len(counter.keys())
    print("average commonality: " + str(average_commonality))
    delineator = "-" * 34
    print(delineator)

#0123456789012345678901234567890123456789012345678901234567890123456789012345678

def main( inagural_path : str, student_path : str, path_to_reporting_verbs : str ):
    """
    Main function to count frequency of reporting verbs in 2 bodies of texts
    and display statistics to the console. Specifically meant for Duncan 
    Buell's PP1 in CS401 at Denison University, where we find the top 20 
    frequencies of reporting verbs in student texts and inagural addresses, and
    compare the words and frequencies across both corpus'.
    
    Args:
        * path_to_corpus_1 (str): the path to the inagural addresses to compare,
        * path_to_corpus_2 (str): the path to the student essays to compare
        * path_to_reporting_verbs (str): the path to a file containing a 
        reporting verb on each line
    Returns:
        nothing
    """

    # grab the paths to inagural addresses
    file_names_inagural = glob_file_names( inagural_path, "sentpos" )
    # grab the paths to final drafts of student papers
    file_names_student = glob_file_names( student_path, "sentpos*final" )

    # calculate and print tables for the 20 most common reporting verbs in
    # first 10 texts in each corpus
    counts_inagural, tokens_inagural = get_reporting_verb_count( file_names_inagural[:10], path_to_reporting_verbs )
    print("Inagural Address Reporting Verb Frequencies:")
    print_formatted_frequencies( counts_inagural, tokens_inagural, 20 )
    counts_student, tokens_student = get_reporting_verb_count( file_names_student[:10], path_to_reporting_verbs )
    print("Student Essay Reporting Verb Frequencies:")
    print_formatted_frequencies( counts_student, tokens_student, 20 )
    print("\n")

    # analyze and compare results of top 20 reporting verbs across both corpus'
    # 01) find top 20 for both texts
    top_20_inagural_verbs = counts_inagural.most_common( 20 )
    top_20_student_verbs = counts_student.most_common( 20 )

    # 02) find & print the average word lengths of each text
    inagural_avg_len = find_avg_word_lengths( top_20_inagural_verbs )
    student_avg_len = find_avg_word_lengths( top_20_student_verbs )
    print("average inagural address reporting verb length: " + str(inagural_avg_len))
    print("average student essay reporting verb length: " + str(student_avg_len))
    print("\n")

    # 03) find + print the overlapping words between the 2 texts
    same_words = find_same_words( top_20_inagural_verbs, top_20_student_verbs )
    print("shared words:")
    print(same_words)

main(
     "./sentposnltk/sentposinauguralnltk", 
     "./sentposnltk/sentposstudent2014nltk",
     "./reporting_verbs.txt"
)
