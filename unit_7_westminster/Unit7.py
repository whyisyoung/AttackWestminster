# Author: Colm Gallagher
# Unit 7
# This script reads a plaintext file of articles named "articles.txt" and
# generates 5 types of summaries for them.
#
# 1. cluster_summary.txt - Our final used summary method which clusters the
#       articles, then summaries each article in the cluster, finally
#       summarizing the concatenated cluster summaries.
# 2. summary.txt - Similar to cluster_summary, but it summarizes the documents
#       before clustering.
# 3. sents_summary.txt - clusters sentences of summarized documents.
# 4. bigrams_summary.txt - clusters two sentences of summarized documents.
# 5. trigrams_summary.txt - clusters three sentences of summarized documents.
#
# Note that this script is not a direct function, and will not reproduce the
# same summary on each run, due to the regression-like nature of the clustering.

from gensim.summarization.summarizer import summarize
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import ngrams
from string import punctuation
import numpy as np
from collections import defaultdict
from heapq import nlargest
import tqdm

filter_phrases = ['photo', 'read more', 'image', 'map']

def filter_document(text):
    sentences = sent_tokenize(text)
    new_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        remove_sent = False
        if ':' in sent_lower[:20]:
            continue
        for phrase in filter_phrases:
            if phrase in sent_lower:
                remove_sent = True
                break
        if not remove_sent:
            new_sentences.append(sent)
    return '\n'.join(new_sentences)

def summarize_text(text, ratio = 0.05):
    sentences = sent_tokenize(text)
    sentences = list(filter(lambda x: len(word_tokenize(x)) < 50, sentences))
    text = '\n'.join(sentences)
    if len(sentences) > 1:
        return ' '.join(summarize(text, ratio=ratio, split=True))
    return ""

def filterDuplicates(data, eps = 0.9, verbose = False):
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
    X = vectorizer.fit_transform(data)
    pairwise_similarity = X * X.T
    toRemove = set()
    for i, pair in enumerate(pairwise_similarity.A):
        if i not in toRemove:
            for j, p in enumerate(pair):
                if i != j and p > eps:
                    toRemove.add(j)
    toRemove = list(toRemove)
    toRemove.sort(reverse=True)
    for i in toRemove:
        p = data.pop(i)
    if verbose:
        print('Removed:', len(toRemove))

def kmeans_cluster(samples, n_clusters, n_init=10, verbose=False):
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
    X = vectorizer.fit_transform(samples)
    km = KMeans(n_clusters = n_clusters, max_iter = 100, n_init = n_init, verbose = False)
    km.fit(X)
    np.unique(km.labels_, return_counts=True)

    text={}
    for i,cluster in enumerate(km.labels_):
        oneDocument = samples[i]
        if cluster not in text.keys():
            # text[cluster] = oneDocument
            text[cluster] = [oneDocument]
        else:
            # text[cluster] += oneDocument + '\n'
            text[cluster].append(oneDocument)
    if verbose:
        stopWords = set(stopwords.words('english')+list(punctuation))
        keywords = {}
        counts={}
        for cluster in range(n_clusters):
            word_sent = word_tokenize(('\n'.join(text[cluster])).lower())
            word_sent=[word for word in word_sent if word not in stopWords]
            freq = FreqDist(word_sent)
            keywords[cluster] = nlargest(100, freq, key=freq.get)
            counts[cluster]=freq
        uniqueKeys={}
        for cluster in range(n_clusters):
            other_clusters=list(set(range(n_clusters))-set([cluster]))
            keys_other_clusters=set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
            unique=set(keywords[cluster])-keys_other_clusters
            uniqueKeys[cluster]=nlargest(10, unique, key=counts[cluster].get)
        print(uniqueKeys)
    return text

def summarize_cluster(cluster):
    items = list(cluster.items())
    # items.sort(key=lambda x: len(x[1]), reverse=True)
    summary = ""
    for i, text in items:
        summary += summarize_text(' '.join(text), ratio=0.01) + '\n\n'
    return summary

f = open('articles.txt', 'r')
texts = [line for line in f]
f.close()
filterDuplicates(texts, verbose=True)
texts = list(map(filter_document, texts))
filterDuplicates(texts, verbose=True)

text_clusters = kmeans_cluster(texts, 5, verbose=True)
cluster_summaries = {}
for i, ctexts in text_clusters.items():
    cluster_summaries[i] = []
    for x in tqdm.tqdm(map(summarize_text, ctexts), total=len(ctexts)):
        cluster_summaries[i].append(x)
    filterDuplicates(cluster_summaries[i], verbose=True)

cluster_summary = ""
cluster_items = list(cluster_summaries.items())
cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
for i, stexts in cluster_items:
    text = '\n'.join(stexts)
    sentences = sent_tokenize(text)
    filterDuplicates(sentences)
    cluster_summary += summarize_text('\n'.join(sentences), ratio=0.02) + '\n\n'
with open('cluster_summary.txt', 'w') as f:
    f.write(cluster_summary)

summaries = []
for x in tqdm.tqdm(map(summarize_text, texts), total=len(texts)):
    summaries.append(x)
filterDuplicates(summaries, verbose=True)
summary_clusters = kmeans_cluster(summaries, 5, verbose=True)
summary_sents = sent_tokenize('\n'.join(summaries))
filterDuplicates(summary_sents, verbose=True)
sents_clusters = kmeans_cluster(summary_sents, 5, verbose=True)
bigrams = list(map(lambda x: ' '.join(x), ngrams(summary_sents, 2)))
bigram_clusters = kmeans_cluster(bigrams, 5, verbose=True)
trigrams = list(map(lambda x: ' '.join(x), ngrams(summary_sents, 3)))
trigram_clusters = kmeans_cluster(trigrams, 5, verbose=True)

summary = summarize_cluster(summary_clusters)
with open('summary.txt', 'w') as f:
    f.write(summary)
sents_summary = summarize_cluster(sents_clusters)
with open('sents_summary.txt', 'w') as f:
    f.write(sents_summary)
bigrams_summary = summarize_cluster(bigram_clusters)
with open('bigrams_summary.txt', 'w') as f:
    f.write(bigrams_summary)
trigrams_summary = summarize_cluster(trigram_clusters)
with open('trigrams_summary.txt', 'w') as f:
    f.write(trigrams_summary)
