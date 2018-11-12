# encoding=utf8
# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import traceback
import io

from os import listdir
from os.path import isfile, join

from gensim.summarization.summarizer import summarize
from timeit import default_timer as timer


def summarize_all_summaries():
    mypath = 'result'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[0] != '.']

    start = timer()
    text = ''
    sentences = set()
    # name = '0.txt'
    for name in onlyfiles:
        with open(mypath + '/' + name, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    line = line.strip()
                    if line not in sentences:
                        sentences.add(line)
                        text += line + ' '
                except:
                    print(name)
                    print(str(idx) + '\n' + traceback.format_exc())
                    pass
    end = timer()
    print('combine time: ' + str(end - start))
    print(len(sentences))

    # with open('mid-summary.txt', 'w') as fout:
    #     fout.write(text)

    # text = ''
    # with open('mid-summary.txt', 'r') as f:
    #     for line in f:
    #         text += line.strip('\n') + ' '
    # print(text)
    s2 = timer()
    output = summarize(text, word_count=1000)
    e2 = timer()
    print('summarize time: ' + str(e2 - s2))
    with open('final/summary.txt', 'w') as fout:
        fout.write(output)


if __name__ == "__main__":
    start = timer()
    summarize_all_summaries()
    end = timer()
    print('time: ' + str(end - start))
