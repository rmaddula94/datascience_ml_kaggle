# Databricks notebook source
# MAGIC %sh
# MAGIC export KAGGLE_USERNAME=rmaddula
# MAGIC export KAGGLE_KEY=70f2cc66c2aff7b361c8c0ddb48884a5
# MAGIC kaggle competitions download -c house-prices-advanced-regression-techniques

# COMMAND ----------

# MAGIC %sh unzip house-prices-advanced-regression-techniques

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/data_description.txt","/FileStore/data_description.txt")
dbutils.fs.cp("file:/databricks/driver/sample_submission.csv","/FileStore/sample_submission.csv")
dbutils.fs.cp("file:/databricks/driver/test.csv","/FileStore/test.csv")
dbutils.fs.cp("file:/databricks/driver/train.csv","/FileStore/train.csv")

# COMMAND ----------

train_df = spark.read.format("csv").option("header", "true").load("file:/databricks/driver/train.csv")
display(train_df.limit(10))

# COMMAND ----------

display(train_df.describe())

# COMMAND ----------

for column in train_df.columns:
  if len(train_df.filter(train_df[column].isNull()).collect()) > 0:
    print(column)

# COMMAND ----------

NAColumns =[]
for column in train_df.columns:
  if len(train_df.filter(train_df[column].isin('?','NaN','NA')).collect()) > 0:
    NAColumns.append(column)

# COMMAND ----------

print(NAColumns)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from functools import reduce
from operator import add
from pyspark import pandas as pd
NA_cnt = train_df.select([count(when(train_df[column].isin('?','NaN','NA'), column)).cast(IntegerType()).alias(column) for column in NAColumns])
NA_sum = NA_cnt.withColumn("Total" ,reduce(add, [col(x) for x in NA_cnt.columns]))
NA_cnt_pct = NA_sum.select([((100*NA_sum[col])/NA_sum['Total']).alias(col) for col in NAColumns])
NA_cnt_pct = NA_cnt_pct.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
ax = NA_cnt_pct.plot.bar(rot=0)
