import os
os.environ['PYSPARK_PYTHON'] = './pyspark/external_pkgs/bin/python'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('unit2').getOrCreate()


import ast
import nltk
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.corpus import words
from nltk.corpus import state_union
from timeit import default_timer as timer
import string


# def get_freq_words_from_unit1(filename):
# 	with open(filename, 'r') as f:
# 		for line in f:
# 			line_dict = ast.literal_eval(line.strip())


def get_synsets(word):
	from nltk.corpus import wordnet as wn
	return set(x.name() for x in wn.synsets(word))


# def get_lemmas(word):
# 	from nltk.corpus import wordnet as wn
# 	synsets = wn.synsets(word)
# 	lemmas = []
# 	for s in synsets:
# 		lemmas += s.lemma_names()
# 	return set(lemmas)


def get_lemma_set(words):
	from nltk.corpus import wordnet as wn
    wordnet_synsets = [(word, set(s for s in wn.synsets(word))) for word in words]
    return [(x[0], list(set(lemma.lower()) for synset in x[1] for lemma in synset.lemma_names())) for x in wordnet_synsets]


def get_percent_usage(words, event_tokens, baseline_tokens, key='diff'):
    word_usage = []
    for word in words:
        baseline_usage = (float) (100.0 * baseline_tokens.count(word) / len(baseline_tokens))
        event_usage = (float) (100.0 * event_tokens.count(word) / len(event_tokens))
        word_usage.append({'word': word, 'event': event_usage, 'baseline': baseline_usage, 'diff': event_usage - baseline_usage})
    return sorted(word_usage, key=lambda x: x[key], reverse=True)


def get_lemma_percent_usage(lemma_map, event_tokens, baseline_tokens, key='diff'):
    lemma_usage = []
    for word, lemmas in lemma_map:
        baseline_count = 0
        event_count = 0
        for lemma in lemmas:
            baseline_count += baseline_tokens.count(lemma)
            event_count += event_tokens.count(lemma)
        baseline_usage = (float) (100.0 * baseline_count / len(baseline_tokens))
        event_usage = (float) (100.0 * event_count / len(event_tokens))
        lemma_usage.append({'word': word, 'lemmas': lemmas, 'event': event_usage, 'baseline': baseline_usage, 'diff': event_usage - baseline_usage})
    return sorted(lemma_usage, key=lambda x: x[key], reverse=True)


def get_lemma_diff(lemma_usage, key='diff'):
    return sorted(lemma_usage, key=lambda x: x[key], reverse=True)


def main():
	start = timer()

	tokensDF = spark.read.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_tokenized.json")
	freqTokensDF = spark.read.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit1-words.json")
	
	custom_stopwords = ["``", "''", "'s", "said", "could", "also", "news", "--", "...", "``", "''"]
	stop_words = set(stopwords.words('english') + list(string.punctuation) + custom_stopwords)

	# top_wordnet = set(freqTokensDF.rdd.flatMap(lambda x: get_synsets(x.word)).collect())
	# counter = Counter(tokensDF.rdd.flatMap(lambda x: x.tokens).flatMap(get_synsets).filter(lambda x: x in top_wordnet).collect())

	filtered_words = tokensDF.rdd.flatMap(lambda x: x.tokens_lower).collect() 

	# print("length of filtered_words: ") 
	# print(len(filtered_words))
	# print(filtered_words[0:10])
	# freq_synset_count = counter.most_common(100)
	# freq_synset = [word for word, _ in freq_synset_count]
	freq_words = freqTokensDF.rdd.map(lambda x: x.word).collect()
	
	# print("top 10 words from unit 1:")
	# print(freq_words[0:10])

	baseline_lower = [x.lower() for x in nltk.Text(brown.words() + words.words() + state_union.words()) if x.lower() not in stop_words]

	word_usage = get_percent_usage(freq_words, filtered_words, baseline_lower, key="event")

	# print("most_common type: " + str(type(freq_synset_count))). # list
	# print("word_usage type: " + str(type(word_usage)))  # list 
	# countDF = spark.createDataFrame(freq_synset_count, ['synset', 'count'])
	# countDF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit2_word_count_big.json", mode="overwrite")

	# count_usage_DF = spark.createDataFrame(word_usage, ['word', 'event', 'baseline', 'diff'])
	count_usage_DF = spark.createDataFrame(word_usage)

	count_usage_DF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit2_word_freq_big.json", mode="overwrite")
	# count_usage_DF.write.csv("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit2_word_freq_big.csv", mode="overwrite")

	# lemma_wordnet = set(freqTokensDF.rdd.flatMap(lambda x: get_lemmas(x.word)).collect())
	# lemma_counter = Counter(tokensDF.rdd.flatMap(lambda x: x.tokens).flatMap(get_lemmas).filter(lambda x: x in lemma_wordnet).collect())
	lemmas = get_lemma_set(freq_words)
	# print('lemmas: ' + str(type(lemmas)))  # list
	# print('lemmas[0]: ' + str(type(lemmas[0])))  # tuple

	lemma_usage = get_lemma_percent_usage(lemmas, filtered_words, baseline_lower, key='event')
	# print('lemma_usage: ' + str(type(lemma_usage)))  # list

	# lemma_countDF = spark.createDataFrame(lemma_counter.most_common(100), ['lemma', 'count'])
	# TODO: Error: TypeError: not supported type: <type 'set'>
	lemma_usage_DF = spark.createDataFrame(lemma_usage)
	lemma_usage_DF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit2_lemma_freq_big.json", mode="overwrite")

	lemma_diff = get_lemma_diff(lemma_usage, key='diff')
	lemma_diff_DF = spark.createDataFrame(lemma_diff)
	lemma_diff_DF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit2_final_result.json", mode="overwrite")

	end = timer()

	print('time elapsed: ' + str(end - start))


if __name__ == '__main__':
	main()
