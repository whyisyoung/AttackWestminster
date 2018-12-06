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

from gensim.summarization.summarizer import summarize
from timeit import default_timer as timer

# INPUT_FILE = 'cleaned_text_westminster.txt'
# INPUT_FILE = 'minimized_cleaned_text_westminster.txt'
INPUT_FILE = 'big_cleaned_filter_3_gram.txt'
# INPUT_FILE = 'tmp-200.txt'


def is_no_ending_mark(text):
    m = re.search(r'[.?!]$', text)
    if m is not None:
        return False
    return True


def get_documents(filename):
    candidates = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if is_no_ending_mark(line):
                line = line + '.'
            candidates.append(line)
    return candidates


def split(candidates, process_total, idx):
    each_proc = len(candidates) / process_total
    documents = candidates[idx * each_proc:(idx+1) * each_proc]
    return documents


def summarize_document(process_total, process_id):
    candidates = get_documents(INPUT_FILE)

    documents = split(candidates, process_total, process_id)

    with open('step2_result_big_filter_trigram_big/' + str(process_id) + '.txt', 'w') as fout:
        for idx, doc in enumerate(documents):
            try:
                # output_sentences = summarize(doc)
                output_sentences = summarize(doc, ratio=0.1)
                fout.write(output_sentences)
            except:
                print(str(idx) + '\n')
                print('doc: ' + doc)
                print(traceback.format_exc())
                pass


if __name__ == '__main__':
    start = timer()
    print(sys.argv)
    try:
        process_total = int(sys.argv[1])
        process_id = int(sys.argv[2])
    except:
        print('please indicate two args')
        sys.exit(-1)

    summarize_document(process_total, process_id)
    end = timer()
    print('time: ' + str(end - start))
