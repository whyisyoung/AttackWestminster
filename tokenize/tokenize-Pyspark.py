import os
os.environ['PYSPARK_PYTHON'] = './pyspark/external_pkgs/bin/python'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tokenize').getOrCreate()

cleanDF = spark.read.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_cleaned.json")

import string
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

custom_stopwords = ["``", "''", "'s", "said", "could", "also", "news", "--", "..."]
stop_words = set(stopwords.words('english') + list(string.punctuation) + custom_stopwords)
punctuation = set(["``", "''", "--", "..."] + list(string.punctuation))

def getTokens(text, stopwords):
    from nltk.tokenize import word_tokenize
    word_tokens = word_tokenize(text.encode('ascii','ignore'))
    filtered_words = [w for w in word_tokens if w.lower() not in stopwords]
    return filtered_words
getTokensUDF = udf(lambda x: getTokens(x, stop_words), ArrayType(StringType()))
getTokensLowerUDF = udf(lambda tokens: [token.lower() for token in tokens], ArrayType(StringType()))
getTokensStopwordsUDF = udf(lambda x: getTokens(x, punctuation), ArrayType(StringType()))

tokensDF = cleanDF.withColumn('tokens', getTokensUDF(cleanDF.text))
tokensDF = tokensDF.withColumn('tokens_lower', getTokensLowerUDF(tokensDF.tokens))
tokensDF = tokensDF.withColumn('tokens_with_stopwords', getTokensStopwordsUDF(tokensDF.text))

tokensDF.write.json('/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_tokenized.json', mode="overwrite")
