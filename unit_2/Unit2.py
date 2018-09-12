# -*- coding: utf-8 -*-
from collections import Counter
import json
import string
import timeit
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
# common_words = ['hurricane', 'storm', 'winds', 'power', 'people', 'coast',
#                 'tropical', 'flooding', 'center', 'water', 'island', 'weather',
#                 'surge', 'climate', 'atlantic', 'service', 'damage', 'expected',
#                 'high', 'sea', 'landfall', 'coastal', 'residents']

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

def plot_cond_freq_dist(words, baseline):
    languages = ['English']
    cfdce = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in word_tokens)
    cfdbl = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in baseline_lower)
    cfdce.plot(cumulative=True)
    cfdbl.plot(cumulative=True)

def get_percent_usage(words, event_tokens, baseline_tokens, key='diff'):
    word_usage = []
    for word in words:
        baseline_usage = 100 * baseline_tokens.count(word) / len(baseline_tokens)
        event_usage = 100 * event_tokens.count(word) / len(event_tokens)
        word_usage.append({'word': word, 'event': event_usage, 'baseline': baseline_usage, 'diff': event_usage - baseline_usage})
    return sorted(word_usage, key=lambda x: x[key], reverse=True)

def get_lemma_set(words):
    wordnet_synsets = [(word, set(s for s in wn.synsets(word))) for word in words]
    return [(x[0], set(lemma.lower() for synset in x[1] for lemma in synset.lemma_names())) for x in wordnet_synsets]
    # return set(lemma.lower() for synset in wordnet_synsets for lemma in synset.lemma_names())

def get_lemma_percent_usage(lemma_map, event_tokens, baseline_tokens, key='diff'):
    lemma_usage = []
    for word, lemmas in lemma_map:
        baseline_count = 0
        event_count = 0
        for lemma in lemmas:
            baseline_count += baseline_tokens.count(lemma)
            event_count += event_tokens.count(lemma)
        baseline_usage = 100 * baseline_count / len(baseline_tokens)
        event_usage = 100 * event_count / len(event_tokens)
        lemma_usage.append({'word': word, 'lemmas': lemmas, 'event': event_usage, 'baseline': baseline_usage, 'diff': event_usage - baseline_usage})
    return sorted(lemma_usage, key=lambda x: x[key], reverse=True)

def main():
    print("Start: Initialize Variables")
    start = timeit.default_timer()
    sentences_str = fetch_sentences(INPUT_FILE)
    # TODO: After converting to lowercase, we may not know whether a word is a proper noun.
    word_tokens = word_tokenize(sentences_str.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation) + custom_stopwords)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    freq_words_with_counts = get_most_frequent_words(filtered_words, 500)
    freq_words = [word for word, _ in freq_words_with_counts]
    baseline_lower = [x.lower() for x in nltk.Text(brown.words() + words.words() + state_union.words()) if x.lower() not in stop_words]
    end = timeit.default_timer()
    print("End: Initialize Variables (took: %0.2fs)" % (end - start))

    ### Used to determine the typical length of words for both baseline and our words.
    # print("Start: Plot Conditional Frequency Distributions")
    # start = timeit.default_timer()
    # plot_cond_freq_dist(word_tokens, baseline_lower)
    # end = timeit.default_timer()
    # print("End: Plot Conditional Frequency Distributions (took: %0.2fs)" % (end - start))

    ### Used to find the most common words in our dataset that help identify our dataset.
    print("Start: Word Frequency Usage Calculation")
    start = timeit.default_timer()
    word_usage = get_percent_usage(freq_words, filtered_words, baseline_lower)
    with open('result/unit2_word_freq.csv', 'w') as f:
        for word in word_usage:
            f.write('%s,%.4f\n' % (word['word'], word['event']))
    end = timeit.default_timer()
    print("End: Word Frequency Usage Calculation (took: %0.2fs)" % (end - start))

    ###  Used to find the most frequent words when counting their synonyms as well.
    print("Start: Lemma Frequency Usage Calculation")
    start = timeit.default_timer()
    lemmas = get_lemma_set(freq_words)
    lemma_usage = get_lemma_percent_usage(lemmas, filtered_words, baseline_lower)
    with open('result/unit2_lemma_freq.csv', 'w') as f:
        for lemma in lemma_usage:
            f.write('%s,%.4f\n' % (lemma['word'], lemma['event']))
    end = timeit.default_timer()
    print("End: Lemma Frequency Usage Calculation (took: %0.2fs)" % (end - start))

if __name__ == '__main__':
    main()
