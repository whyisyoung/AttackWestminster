import os
os.environ['PYSPARK_PYTHON'] = './pyspark/external_pkgs/bin/python'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('unit3').getOrCreate()

tokensDF = spark.read.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_tokenized.json")

import nltk
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from collections import Counter
from nltk.corpus import stopwords
import string

# [1] Tag tokens with POS
POSTagUDF = udf(lambda x: nltk.pos_tag(x), ArrayType(ArrayType(StringType())))
posRDD = tokensDF.rdd.flatMap(lambda x: nltk.pos_tag(x.tokens_with_stopwords)).map(lambda x: (x[0].lower(), x[1])).filter(lambda x: x[0] not in stop_words)

custom_stopwords = ["``", "''", "'s", "said", "could", "also", "news", "--", "..."]
stop_words = set(stopwords.words('english') + list(string.punctuation) + custom_stopwords)

# [2] Get most frequent nouns
counter = Counter(posRDD.filter(lambda x: x[1][0] == 'N').map(lambda x: x[0]).collect())
countDF = spark.createDataFrame(counter.most_common(100), ['noun', 'count'])
countDF.write.csv('/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_unit3_nouns.csv', mode="overwrite")

# [3] Get most frequent verbs
counter = Counter(posRDD.filter(lambda x: x[1][0] == 'V').map(lambda x: x[0]).collect())
countDF = spark.createDataFrame(counter.most_common(100), ['verb', 'count'])
countDF.write.csv('/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_unit3_verbs.csv', mode="overwrite")
