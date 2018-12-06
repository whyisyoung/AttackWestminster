# This can be run using the following command on the Hadoop cluster:
# spark2-submit --archives external_pkgs.zip#pyspark,nltk_data.zip#nltk_data
#   --conf spark.yarn.appMasterEnv.NLTK_DATA=./nltk_data/
#   --conf spark.executorEnv.NLTK_DATA=./nltk_data/ SCRIPT_NAME.py
import os
os.environ['PYSPARK_PYTHON'] = './pyspark/external_pkgs/bin/python'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cleanText').getOrCreate()

# This file needs to be put into HDFS in advance (or already be there from an
# existing script).
htmlDF = spark.read.json("/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/html_sentences.json")

def htmlClean(htmlText):
    import justext
    paragraphs = justext.justext(htmlText, justext.get_stoplist("English"))
    return ' '.join([p.text for p in paragraphs if not p.is_boilerplate])

# Filters out empty texts before and after cleaning with jusText.
cleanedDF = htmlDF.rdd.filter(lambda x: x.text != '').map(lambda x: (x.title, x.originalUrl, htmlClean(x.text))).filter(lambda x: x[2] != '').toDF() \
    .withColumnRenamed("_1", "title").withColumnRenamed("_2", "originalUrl").withColumnRenamed("_3", "text")

# Writes to HDFS. Be careful this overwrites existing results.
cleanedDF.write.json('/user/cs4984cs5984f18_team4/4_Attack_Westminster_big/Attack_Westminster_big_cleaned.json', mode="overwrite")
