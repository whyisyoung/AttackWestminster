# This can be run using the following command on the Hadoop cluster:
#spark2-submit --archives external_pkgs.zip#pyspark,nltk_data.zip#nltk_data --conf spark.yarn.appMasterEnv.NLTK_DATA=./nltk_data/ --conf spark.executorEnv.NLTK_DATA=./nltk_data/ Unit1-Pyspark.py
import os
os.environ['PYSPARK_PYTHON'] = './pyspark/external_pkgs/bin/python'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('unit1').getOrCreate()

# This file needs to be put into HDFS in advance (or already be there from an
# existing script).
cleanedSet = spark.read.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_tokenized.json")

import nltk
from collections import Counter

#Most frequent words
counters = cleanedSet.rdd.map(lambda x: Counter(x.tokens_lower)).collect()
counter = Counter()
for c in counters:
    counter.update(c)
countDF = spark.createDataFrame(counter.most_common(100), ['word', 'count'])
countDF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit1-words.json")

#Most frequent bigrams
counters = cleanedSet.rdd.map(lambda x: Counter(nltk.bigrams(x.tokens_lower))).collect()
counter = Counter()
for c in counters:
    counter.update(c)
countDF = spark.createDataFrame(counter.most_common(100), ['bigram', 'count'])
countDF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit1-bigrams.json")

#Most frequent trigrams
counters = cleanedSet.rdd.map(lambda x: Counter(nltk.trigrams(x.tokens_lower))).collect()
counter = Counter()
for c in counters:
    counter.update(c)
countDF = spark.createDataFrame(counter.most_common(100), ['trigram', 'count'])
countDF.write.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Unit1-trigrams.json")
