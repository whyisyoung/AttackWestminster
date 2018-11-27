# -*- coding: utf-8 -*-
# @Author: liminyang
# @Date:   2018-10-24 11:18:36
# @Last Modified by:   liminyang
# @Last Modified time: 2018-11-04 01:13:03
import os
import sys
from nltk.corpus import stopwords
import nltk.data
import traceback
import multiprocessing
import re
import time
import ast

from tqdm import tqdm

from gensim.summarization.summarizer import summarize
from timeit import default_timer as timer


# INPUT_FILE = 'cleaned_text_westminster.txt'
# INPUT_FILE = 'minimized_cleaned_text_westminster.txt'

# INPUT_FILE = 'big_data/big_cleaned.json'
UNIT1_UNIGRAM_FILE = '/Users/liminyang/Google Drive/course/cs-5984/AttackWestminster/unit_1_westminster/big-result/Unit1-words.json'
UNIT1_BIGRAM_FILE = '/Users/liminyang/Google Drive/course/cs-5984/AttackWestminster/unit_1_westminster/big-result/Unit1-bigrams.json'
UNIT1_TRIGRAM_FILE = '/Users/liminyang/Google Drive/course/cs-5984/AttackWestminster/unit_1_westminster/big-result/Unit1-trigrams.json'


def get_gram(option):
    keywords = []
    if option == 1:
        with open(UNIT1_UNIGRAM_FILE, 'r') as f:
            for line in f:
                try:
                    keywords.append(ast.literal_eval(line[:-1])['word'])
                except:
                    print(traceback.format_exc())
    elif option == 2:
        with open(UNIT1_BIGRAM_FILE, 'r') as f:
            for line in f:
                try:
                    line_dict_bigram = ast.literal_eval(line[:-1])['bigram']
                    _first = line_dict_bigram['_1']
                    _second =line_dict_bigram['_2']
                    keywords.append(_first + ' ' + _second)
                except:
                    print(traceback.format_exc())
    elif option == 3:
        with open(UNIT1_TRIGRAM_FILE, 'r') as f:
            for line in f:
                try:
                    line_dict_trigram = ast.literal_eval(line[:-1])['trigram']
                    _first = line_dict_trigram['_1']
                    _second = line_dict_trigram['_2']
                    _third = line_dict_trigram['_3']
                    keywords.append(_first + ' ' + _second + ' ' + _third)
                except:
                    print(traceback.format_exc())
    return keywords

def is_relevant(text, option):
    keywords = get_gram(option)
    for key in keywords:
        if key in text:
            return True
    return False


def get_relevant_docs(input_file, output_file, option):
    with open(output_file, 'w') as fout:
        with open(input_file, 'r') as f:
            # for line in tqdm(f):  # no progress percent bar, need to specify total to tell tqdm the total iterations
            content = f.readlines()
            for idx in tqdm(range(len(content))):
                line = content[idx]
                line_dict = ast.literal_eval(line[:-1])
                text = line_dict['text']
                if is_relevant(text, option):
                    fout.write(text + '\n')
                    # fout.write(text.encode('utf-8') + '\n')


# def write_text_to_file(json_file, txt_file):
#     with open(txt_file, 'w') as fout:
#         with open(json_file, 'r') as fin:
#             for idx, line in enumerate(fin):
#                 try:
#                     line_dict = ast.literal_eval(line[:-1])
#                     text = line_dict['text']
#                     if '\n' in text:
#                         print(str(idx + 1))
#                     fout.write(text + '\n')
#                 except:
#                     print(traceback.format_exc())


if __name__ == '__main__':
    # =======================
    # for test
    # write_text_to_file('big_data/big_cleaned.json', 'big_data/big_cleaned_original.txt')
    # =======================

    start = timer()
    print(sys.argv)
    try:
        INPUT_FILE = sys.argv[1]
        option = int(sys.argv[2])
        OUTPUT_FILE = 'big_cleaned_filter_' + str(option) + '_gram.txt'
    except:
        print('please indicate input and option')
        sys.exit(-1)

    get_relevant_docs(INPUT_FILE, OUTPUT_FILE, option)
    end = timer()
    print('time: ' + str(end - start))
