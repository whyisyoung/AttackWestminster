import sys
reload(sys)
sys.setdefaultencoding('utf8')
import nltk
#from nltk.book import *
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import json
import time
from multiprocessing import Pool
from collections import Counter
from nltk.stem.porter import PorterStemmer
import pickle
import gensim
from gensim import corpora, models
import numpy as np

input_file_path = "Preprocess/Attack_Westminster_big_cleaned.json"
# input_file_path = "Preprocess/small-cleaned.json"

output_path = "unit_6/big"

def get_most_frequent_words():
    docfreq = pickle.load(open(picklefile, 'r'))
    max_f = docfreq.most_common(1)[0][1]
    print max_f
    stop_words = [k for k in docfreq if docfreq[k] > 0.5*max_f]
    print stop_words
    return stop_words

def get_stopwords():
    print "Retrieving list of stopwords..."
    nltk_defaults = stopwords.words('english')
    punctuation = [',', ':', '.', '...', ';', '-', '"', '--', '!', '?', '(', ')', '``', '\'\'']
    custom_words = ['would', 'though', 'it', 'still', 'he', 'at', 'even', 'but', 'like', 'upon', 'a', 'mr.']
    return set(nltk_defaults + punctuation + custom_words)

def get_input(fname):
    stop_words = get_stopwords()
    print "Reading Input JSON..."
    with open(fname) as f:
        content = f.readlines()
    content = [(json.loads(x.lower()), stop_words) for x in content]
    print "Done loading json!"
    return content

def test():
    documents = get_input(input_file_path)
    print documents


def evaluate(row):
    # clean data
    record, stop_words = row
    # text = nltk.word_tokenize(record['sentences_t'].lower())
    text = nltk.word_tokenize(record['text'].lower())
    p_stemmer = PorterStemmer()
    stemmed_stopwords = [p_stemmer.stem(i) for i in stop_words]
    stemmed_text = [p_stemmer.stem(i) for i in text]
    stopped_tokens = [i for i in stemmed_text if (i not in stemmed_stopwords and len(i) > 2)]
    return stopped_tokens

def appendArticles(listOfAricles, output_file, documents):
    with open(output_path + output_file + ".txt", 'w') as f:
        for article in listOfAricles:
            f.write(documents[article][0]['text'])
            # f.write(documents[article][0]['sentences_t'])


def main():
    st = time.time()
    print "Start Time: ", st

    documents = get_input(input_file_path)

    p = Pool(15)
    #urls = [row[0]['URL_s'] for row in documents]
    individual_results = p.map(evaluate, documents)
    dictionary = corpora.Dictionary(individual_results)
    corpus = [dictionary.doc2bow(text) for text in individual_results]
    tfidf = gensim.models.TfidfModel(corpus)
    imp_corpus = tfidf[corpus]

    #LSA
    lsimodel = models.LsiModel(imp_corpus, id2word = dictionary)

    #LDA
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=imp_corpus,
    #                                        id2word=dictionary,
    #                                        num_topics=6,
    #                                        random_state=100,
    #                                        update_every=1,
    #                                        chunksize=100,
    #                                        passes=10,
    #                                        alpha='auto',
    #                                        per_word_topics=True)

    #cohmodel = models.CoherenceModel(model=lda_model, corpus=imp_corpus, coherence='u_mass')
    cohmodel = models.CoherenceModel(model=lsimodel, corpus=imp_corpus, coherence='u_mass')
    # print 'Coherence:', cohmodel.get_coherence_per_topic()
    lsi_corpus = lsimodel[imp_corpus]
    #lsi_corpus = lda_model[imp_corpus]
    # Use the singular values to choose how many components to use
    v = lsimodel.projection.s**2 / sum(lsimodel.projection.s**2)
    #print v[:100]
    #k = np.argmin(v>0.005)+1    # Hard threshold, may be better to plot and find the knee

    #At the moment just print out 15 topics
    topics = lsimodel.show_topics(num_topics=15, num_words=5)
    #topics = lda_model.show_topics(num_topics=15, num_words=5)
    #topcis2 = ldamodel.get_topics()

    for i, topic in enumerate(topics):
        print topic

    listOfDocsPerTopic = []
    for i, topic in enumerate(topics):
        articles = []
        tops = sorted(zip(range(len(lsi_corpus)), lsi_corpus), reverse=True, key=lambda doc: abs(dict(doc[1]).get(i, 0.0)))

        curr = tops[0][1][i][1]
        j = 0
        while (abs(curr) > 0.3 and j < len(tops)):
            top = tops[j]
            j += 1
            curr = top[1][i][1]
            articles.append(top[0])
        listOfDocsPerTopic.append(articles)

    appendArticles(listOfDocsPerTopic[0], "topic-big-0", documents)
    appendArticles(listOfDocsPerTopic[2], "topic-big-2", documents)
    appendArticles(listOfDocsPerTopic[7], "topic-big-9", documents)

    end = time.time()
    print "End Time: ", end-st

if __name__ == '__main__':
    main()
