# -*- coding: utf-8 -*-
from collections import Counter 
# import json
import ast
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from timeit import default_timer as timer


# TODO: After converting to lowercase, we may not know whether a word is a proper noun.

INPUT_FILE = 'input/sentences.json' 
# OUTPUT_FILE = 'result/most_occur_first_try.txt'
# OUTPUT_FILE = 'result/most_occur_second_try.txt'
OUTPUT_FILE = 'result/most_occur_without_stopword.txt'

# TODO: determine the value of k.
top_k = 1000


# TODO: need to add more stopwords to filter unimportant words.
custom_stopwords = ["'s", "said", "could", "also", "news", "--", "..."]


def fetch_sentences(filename):
    """
    Fetch sentences from cleaned json file.
    """
    sentences_str = ""
    with open(filename, 'r') as f:
        for line in f:
            # Another option is using json loads
            # json_acceptable_str = line[:-1].replace("'",  "\"")
            # line_dict = json.loads(json_acceptable_str)
            line_dict = ast.literal_eval(line[:-1])
            sentences_str += ' ' + line_dict["Sentences"]

    return sentences_str


def get_most_frequent_words(data, k):
    counter = Counter(data)
    most_occur = counter.most_common(k)
    return most_occur


def main():
    start = timer()
    sentences_str = fetch_sentences(INPUT_FILE)
    # need to convert all word to lowercase, 
    # or it would treat Apple and apple as two different tokens
    # replace the "." with " " so that "2012" and "2012." would be treated as the same token.
    cleaned_sentences = sentences_str.lower().replace('.', ' ')
    word_tokens = word_tokenize(cleaned_sentences)
    # word_split = cleaned_sentences.split()  # this is wrong because you should use all the punctuations as delimeter
    
    stop_words = set(stopwords.words('english') + list(string.punctuation) + custom_stopwords) 
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    most_occur = get_most_frequent_words(filtered_tokens, top_k)
    
    with open(OUTPUT_FILE, 'w') as f:
        for (word, occur) in most_occur:
            f.write(word + ',' + str(occur) + '\n')
    end = timer()
    print('time elapsed:' + str(end - start))


if __name__ == '__main__':
    main()
