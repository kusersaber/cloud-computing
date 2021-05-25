#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Word2Vec
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.sql.functions import *
from pyspark.mllib.recommendation import ALS
from time import time


# In[2]:


spark = SparkSession     .builder     .appName("WorkLoad")     .getOrCreate()

start = time()
## select user_id, retweet_id and replyto_id and combine the data
df = spark.read.option('multiline','true').json('tweets.json')
user_retweet = df.select("user_id","retweet_id").filter("retweet_id is not null").rdd.groupByKey().mapValues(list)
user_reply = df.select("user_id","replyto_id").filter("replyto_id is not null").rdd.groupByKey().mapValues(list)
user_interest = user_retweet.union(user_reply).reduceByKey(lambda x,y: x+y).mapValues(list)
df_interest = user_interest.toDF().selectExpr("_1 as user_id","_2 as interest")


# In[3]:


## workload 1

## transform the id to index
indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index").fit(df_interest)
data = indexer.transform(df_interest).select("user_id_index","interest").rdd.map(lambda x :x[1])

## TF-IDF
hashingTF = HashingTF()
tf = hashingTF.transform(data)
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
target_user = tfidf.collect()[36]


# cosine value function for tfidf
def cos_tfidf(row):
    return row[2],row[1].dot(target_user)/(row[1].norm(2) * target_user.norm(2))

## combine the tfidf vector with user_id
df_tfidf = tfidf.zipWithIndex().toDF().selectExpr("_1 as vector","_2 as index")
indexWithUserId = df.select("user_id").rdd.zipWithIndex().toDF().selectExpr("_1 as user_id", "_2 as index")
id_vector = df_tfidf.join(indexWithUserId,"index")
cos_sim = id_vector.rdd.map(cos_tfidf).sortBy(lambda x:-x[1]).collect()
result_tfidf = cos_sim[1:6]
print(result_tfidf)


# In[4]:


##explode array to column
df_explode = df_interest.select("user_id",explode("interest"))


##transform the interest to index
indexer1 = StringIndexer(inputCol="col", outputCol="interest_index",stringOrderType = "frequencyDesc").fit(df_explode)
df_index = indexer1.transform(df_explode).select("user_id","interest_index")     .selectExpr("user_id","cast (interest_index as string) index")     .groupBy("user_id").agg(collect_list("index"))

## work2vec

word2vec = Word2Vec(vectorSize=50, minCount=1, inputCol="collect_list(index)", outputCol="result")
model = word2vec.fit(df_index).transform(df_index).select("user_id","result").rdd

## define cosine function for work2vec

target_wk2v = model.collect()[12][1]

def cos_work2vec(row):
    return row[0],row[1].dot(target_wk2v)/(row[1].norm(2) * target_wk2v.norm(2))

cos_wd2v = model.map(cos_work2vec).sortBy(lambda x: -x[1]).collect()
top = cos_wd2v[1:6]


# In[ ]:


print(top)


# In[ ]:


user_mentions = df.select("user_id","user_mentions").filter("user_mentions is not null")
mention = user_mentions.withColumn("mention",explode("user_mentions")).select("user_id","mention")

# delete unuse data
def mention_user(row):
    return row[0],row[1][0]

# count the mention times
mention_data = mention.rdd.map(mention_user).toDF().groupBy("_1","_2").count()     .selectExpr("_1 as user_id","_2 as mention_id","count as times")

## transform user_id and mention_id to index
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index")             .fit(mention_data) for column in list(set(mention_data.columns)-set(['times'])) ]

pipeline = Pipeline(stages=indexers)
indexer_data = pipeline.fit(mention_data).transform(mention_data)
indexer_rdd = indexer_data.rdd.map(lambda x:(x[4],x[3],x[2]))

# Use ALS to train the data
model = ALS.train(indexer_rdd, 5, 5)
testdata = indexer_rdd.map(lambda x:(x[0],x[1]))
predictions = model.predictAll(testdata).toDF()
users = predictions.sort("user").selectExpr("user as user_id_index", "product as mention_id_index","rating as prediction")


# In[ ]:


new = users.join(indexer_data,"user_id_index").orderBy(col("prediction").desc())     .select("user_id","mention_id").groupBy("user_id") .agg(collect_list(col("mention_id")).alias("recommend"))
new.createOrReplaceTempView("new_final")
user_recommend=spark.sql("""
select user_id,
       recommend[0] as recommend_1,
       recommend[1] as recommend_2,
       recommend[2] as recommend_3,
       recommend[3] as recommend_4,
       recommend[4] as recommend_5
from new_final
""")
user_recommend.sort("user_id").write.csv('recommendation.csv')
end = time()
print("time: ",end-start)

