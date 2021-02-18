#This file trains and tests a random forest binary classifer.
#It uses VectorAssembler to transform dataframe to proper format for Spark.
#It uses Scaler to scale the data
#It uses Pipeline to put the RDD of VectorAssembler, Scaler and RandomForest into a pipeline RDD.  Pipelines take advantage of parallel processing.
#It does a grid search via ParamGridBuilder and then performs cross validation to find the best model and test the data.



'''
Usage:

$ PYSPARK_PYTHON=$(which python) nohup spark-submit PySpark_RF.py hdfs:/user/gs3170/train.csv hdfs:/user/gs3170/test.csv ./rf_model1 > rf_output1.txt
'''


# We need sys to get the command line arguments
import sys


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics


def main(spark, train, test):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    train : string, path to the parquet file of training data

    test : string, path to the parquet file of test data
    '''
    ts = time.time()    
    
    
    train_df = spark.read.csv(train, header=True,inferSchema=True)
    test_df = spark.read.csv(test, header=True,inferSchema=True)
    
    ylabel = 'Crime'

    X_cols= train_df[:-1]

    x_train = train_df[x_cols]
    assembler = VectorAssembler(inputCols=x_cols,outputCol="features")
    scaler = StandardScaler(inputCol = 'features', outputCol = 'scaled_features', withStd=True, withMean=False)
    rf = RandomForestClassifier(numTrees=80, maxDepth = 2, featuresCol='scaled_features', labelCol=ylabel, seed=42,cacheNodeIds=False)

    pipeline = Pipeline(stages=[assembler, scaler, rf])
    grid = ParamGridBuilder().addGrid(rf.numTrees, [100,200,300,400,500]).addGrid(rf.maxDepth, [5,10,17,22,30]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol=ylabel,metricName='areaUnderROC')
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5, parallelism=20)
    cv_model = cv.fit(train_df)
    #best_cv = cv_model.bestModel

    #x_test = test_df[x_cols]
    predictions = cv_model.transform(train_df)
        
    evaluator.evaluate(predictions)

    cv_model.avgMetrics[0]
    
    metric = 'areaUnderROC'
    print(metric,': ', evaluator.setMetricName(metric).evaluate(predictions))
    
    metric = 'areaUnderPR'
    print(metric,': ', evaluator.setMetricName(metric).evaluate(predictions))
    td = time.time()
    print('Time: ', td-ts)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('RF').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]

    # And the location to store the trained model
    test_file = sys.argv[2]

    # Call our main routine
    main(spark, train_file, test_file)

