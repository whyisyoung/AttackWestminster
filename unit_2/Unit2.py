# -*- coding: utf-8 -*-
from collections import Counter
import json
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import words
from nltk.corpus import state_union

INPUT_FILE = 'input/sentences.json'

# TODO: need to add more stopwords to filter unimportant words.
custom_stopwords = ["'s", "said", "could", "also", "news", "--", "..."]

# TODO: these common words were partially selected from frequent words.
#       we can do better by looking at data retrieved in this unit.
common_words = ['hurricane', 'storm', 'winds', 'power', 'people', 'coast',
                'tropical', 'flooding', 'center', 'water', 'island', 'weather',
                'surge', 'climate', 'atlantic', 'service', 'damage', 'expected',
                'high', 'sea', 'landfall', 'coastal', 'residents']

def fetch_sentences(filename):
    """
    Fetch sentences from cleaned json file.
    """
    sentences_str = ""
    with open(filename, 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            sentences_str += ' ' + line_dict["Sentences"]
    return sentences_str

def get_most_frequent_words(words, k):
    counter = Counter(words)
    most_occur = counter.most_common(k)
    return most_occur

def main():
    sentences_str = fetch_sentences(INPUT_FILE)
    # TODO: After converting to lowercase, we may not know whether a word is a proper noun.
    word_tokens = word_tokenize(sentences_str.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation) + custom_stopwords)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    freq_words = get_most_frequent_words(filtered_words, 100)

    baseline_lower = [x.lower() for x in nltk.Text(brown.words() + words.words() + state_union.words())]

    ### Used to determine the typical length of words for both baseline and our words.
    # languages = ['English']
    # cfdce = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in word_tokens)
    # cfdbl = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in baseline_lower)
    # cfdce.plot(cumulative=True)
    # cfdbl.plot(cumulative=True)

    ### Used to find the most common words in our dataset that help identify our dataset.
    word_percentages = []
    for x, occurances in freq_words:
        baseline_percentage = 100 * baseline_lower.count(x) / len(baseline_lower)
        event_percentage = 100 * occurances / len(filtered_words)
        word_percentages.append({'word': x, 'diff': event_percentage - baseline_percentage})
    word_percentages = sorted(word_percentages, key=lambda x: x['diff'], reverse=True)

    with open('result/unit2_word_freq.csv', 'w') as f:
        for x in word_percentages:
            f.write('%s,%.4f\n' % (x['word'], x['diff']))

    ###  Used to find the most common synonyms of our common words that help identify our datasetself.
    ###  Can be used to redefine a better common words set.
    wordnet_synsets = set(s for word in common_words for s in wn.synsets(word))
    wordnet_lemmas = set(lemma.lower() for synset in wordnet_synsets for lemma in synset.lemma_names())

    lemma_percentages = []
    for x in wordnet_lemmas:
        baseline_percentage = 100 * baseline_lower.count(x) / len(baseline_lower)
        event_percentage = 100 * filtered_words.count(x) / len(filtered_words)
        lemma_percentages.append({'lemma': x, 'diff': event_percentage - baseline_percentage})
    lemma_percentages = sorted(lemma_percentages, key=lambda x: x['diff'], reverse=True)

    with open('result/unit2_lemma_freq.csv', 'w') as f:
        for x in lemma_percentages:
            f.write('%s,%.4f\n' % (x['lemma'], x['diff']))

if __name__ == '__main__':
    main()
