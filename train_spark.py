"""PySpark ALS recommendation + interpretable genre profiles on MovieLens Latest Small.

Reproducible run (Spark 3.5+, Java 17/21, Python 3.10+):
    python download_data.py
    spark-submit --master 'local[*]' train_spark.py --data-root data --out outputs
This script is a reproducible implementation provided with the presentation.
Its Spark model results must be measured after executing it, and may differ
from the independent numerical reference shown in the presentation.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType, LongType, StringType

SEED = 42
RANKS = (8, 12)
REGS = (0.08, 0.12)
ITERATIONS = 8
RATING_MIN, RATING_MAX = 0.5, 5.0


def clip(x):
    return F.greatest(F.lit(RATING_MIN), F.least(F.lit(RATING_MAX), x))


def baseline_tables(train):
    global_rating = float(train.agg(F.avg('rating')).first()[0])
    users = train.groupBy('userId').agg(F.sum('rating').alias('usum'), F.count('*').alias('un'))
    movies = train.groupBy('movieId').agg(F.sum('rating').alias('msum'), F.count('*').alias('mn'))
    return global_rating, users, movies


def attach_baseline(df, tables):
    global_rating, users, movies = tables
    j = df.join(users, 'userId', 'left').join(movies, 'movieId', 'left')
    ud = F.when(F.col('un').isNotNull(),
                (F.col('usum') / F.col('un') - F.lit(global_rating)) * F.col('un') / (F.col('un') + F.lit(10.0))
                ).otherwise(F.lit(0.0))
    md = F.when(F.col('mn').isNotNull(),
                (F.col('msum') / F.col('mn') - F.lit(global_rating)) * F.col('mn') / (F.col('mn') + F.lit(10.0))
                ).otherwise(F.lit(0.0))
    return j.withColumn('baseline', clip(F.lit(global_rating) + ud + md))


def evaluate(df, label='rating', pred='prediction'):
    n=df.count()
    if n==0: return {'count':0,'rmse':None,'mae':None}
    return {
        'count':n,
        'rmse':float(RegressionEvaluator(metricName='rmse', labelCol=label, predictionCol=pred).evaluate(df)),
        'mae':float(RegressionEvaluator(metricName='mae', labelCol=label, predictionCol=pred).evaluate(df))
    }


def fit_als(df, rank, reg, rating_col):
    est=ALS(userCol='userId',itemCol='movieId',ratingCol=rating_col,rank=rank,regParam=reg,
            maxIter=ITERATIONS,seed=SEED,implicitPrefs=False,coldStartStrategy='drop',
            nonnegative=False)
    return est.fit(df)


def predict_residual(model, base_df):
    predicted=model.transform(base_df.withColumn('residual', F.lit(0.0)))
    return predicted.withColumn('final_prediction',clip(F.col('baseline')+F.col('prediction')))


def make_profiles(ratings, movies, out:Path):
    df=(ratings.join(movies,'movieId','inner')
        .withColumn('genre',F.explode(F.split(F.col('genres'),r'\|')))
        .filter(F.col('genre')!='(no genres listed)'))
    prof=(df.groupBy('userId','genre')
          .agg(F.count('*').alias('rated_count'),F.round(F.avg('rating'),3).alias('avg_rating'),
               F.round(F.avg(F.when(F.col('rating')>=4,F.lit(1.0)).otherwise(F.lit(0.0))),3).alias('positive_ratio')))
    prof.orderBy('userId',F.desc('positive_ratio')).coalesce(1).write.mode('overwrite').option('header',True).csv(str(out/'genre_profiles'))
    return prof


def make_recommendations(spark, model, train, tables, movies, out:Path, top_movies=300):
    # Predict among frequently rated movies. Excludes already rated films; no future/test ratings are used.
    all_users=train.select('userId').distinct()
    candidate_movies=(train.groupBy('movieId').count().orderBy(F.desc('count')).limit(top_movies)
                      .select('movieId'))
    seen=train.select('userId','movieId').distinct()
    pairs=(all_users.crossJoin(F.broadcast(candidate_movies))
           .join(seen,['userId','movieId'],'left_anti'))
    with_base=attach_baseline(pairs,tables)
    pred=predict_residual(model,with_base)
    w=Window.partitionBy('userId').orderBy(F.desc('final_prediction'),F.asc('movieId'))
    result=(pred.withColumn('position',F.row_number().over(w)).filter(F.col('position')<=10)
            .join(movies,'movieId','left')
            .select('userId','position','movieId','title','genres',
                    F.round('final_prediction',3).alias('estimated_rating')))
    result.orderBy('userId','position').coalesce(1).write.mode('overwrite').option('header',True).csv(str(out/'recommendations'))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='data')
    p.add_argument('--out',default='outputs')
    p.add_argument('--master',default=None)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True, exist_ok=True)
    builder=(SparkSession.builder.appName('MovieLens-Spark-ALS-UserProfiles')
           .config('spark.sql.shuffle.partitions','8')
           .config('spark.driver.memory','4g'))
    # Do not override --master supplied by spark-submit.
    if args.master is not None:
        builder=builder.master(args.master)
    spark=builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    base=Path(args.data_root)/'ml-latest-small'
    if not base.exists(): raise FileNotFoundError('Dataset missing. Run: python download_data.py')
    rating_schema=StructType([StructField('userId',IntegerType()),StructField('movieId',IntegerType()),
         StructField('rating',FloatType()),StructField('timestamp',LongType())])
    movie_schema=StructType([StructField('movieId',IntegerType()),StructField('title',StringType()),
        StructField('genres',StringType())])
    ratings=(spark.read.option('header',True).schema(rating_schema).csv(str(base/'ratings.csv'))
             .na.drop(subset=['userId','movieId','rating']).dropDuplicates(['userId','movieId'])
             .filter(F.col('rating').between(RATING_MIN,RATING_MAX)).cache())
    movies=(spark.read.option('header',True).schema(movie_schema).csv(str(base/'movies.csv'))
            .na.drop(subset=['movieId']).dropDuplicates(['movieId']).cache())
    count=ratings.count();movie_count=movies.count(); user_count=ratings.select('userId').distinct().count()
    print(f'MovieLens: {count} ratings, {user_count} users, {movie_count} movies')
    split=ratings.randomSplit([0.8,0.1,0.1],seed=SEED)
    train,valid,test=(part.cache() for part in split)
    sizes=[part.count() for part in (train,valid,test)]
    g,u,i=baseline_tables(train)
    tables=g,u,i
    valid_base=attach_baseline(valid,tables).cache()
    # Train-only baseline metrics, no validation/test leakage
    valid_cold=valid_base.filter(F.col('un').isNotNull() & F.col('mn').isNotNull()).cache()
    base_val=evaluate(valid_cold.withColumn('prediction',F.col('baseline')))
    experiments=[]
    for rank in RANKS:
        for reg in REGS:
            st=time.perf_counter()
            model=fit_als(train,rank,reg,'rating')
            pred=model.transform(valid).withColumn('prediction',clip(F.col('prediction')))
            m=evaluate(pred)
            experiments.append({'variant':'raw_als','rank':rank,'reg':reg,'seconds':round(time.perf_counter()-st,3), **m})
            print('raw ALS',experiments[-1])
            st=time.perf_counter()
            train_res=(attach_baseline(train,tables).withColumn('residual',F.col('rating')-F.col('baseline'))
                        .select('userId','movieId','rating','residual'))
            residual=fit_als(train_res,rank,reg,'residual')
            pred=predict_residual(residual,valid_base)
            m=evaluate(pred,pred='final_prediction')
            experiments.append({'variant':'bias_aware_als','rank':rank,'reg':reg,
                                'seconds':round(time.perf_counter()-st,3), **m})
            print('bias-aware ALS', experiments[-1])
    best=min((e for e in experiments if e['variant']=='bias_aware_als' and e['rmse'] is not None),
             key=lambda e:e['rmse'])
    # Refit on training+validation for final test; test is touched only here.
    trainval=train.unionByName(valid).cache();trainval.count()
    tv_tables=baseline_tables(trainval)
    trainval_res=(attach_baseline(trainval,tv_tables)
                  .withColumn('residual',F.col('rating')-F.col('baseline'))
                  .select('userId','movieId','rating','residual'))
    final_model=fit_als(trainval_res,best['rank'],best['reg'],'residual')
    final_model.write().overwrite().save(str(out/'als_residual_model'))
    test_base=attach_baseline(test,tv_tables)
    test_pred=predict_residual(final_model,test_base).cache()
    test_metrics=evaluate(test_pred,pred='final_prediction')
    test_bias=evaluate(test_pred.withColumn('prediction',F.col('baseline')))
    test_coverage=test_metrics['count']/sizes[2] if sizes[2] else 0
    make_profiles(trainval,movies,out)
    recs=make_recommendations(spark,final_model,trainval,tv_tables,movies,out)
    top_rated=(ratings.groupBy('movieId').agg(F.count('*').alias('rating_count'),
              F.round(F.avg('rating'),3).alias('mean_rating')).join(movies,'movieId')
              .orderBy(F.desc('rating_count')).limit(30))
    top_rated.coalesce(1).write.mode('overwrite').option('header',True).csv(str(out/'popular_movies'))
    hist=[{'rating':float(row['rating']),'count':int(row['count'])}
          for row in ratings.groupBy('rating').count().orderBy('rating').collect()]
    summary={
        'source':'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'engine':'Apache Spark MLlib ALS (PySpark)',
        'seed':SEED,'data':{'ratings':count,'users':user_count,'movies':movie_count,
                           'train':sizes[0],'validation':sizes[1],'test':sizes[2]},
        'rating_histogram':hist,'validation_bias_baseline':base_val,
        'hyperparameters':experiments,'best':best,'test':test_metrics,
        'test_bias_baseline_same_rows':test_bias,
        'test_coverage':test_coverage,
        'recommendation_users':recs.select('userId').distinct().count()
    }
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULTS:',json.dumps(summary,ensure_ascii=False,indent=2))
    spark.stop()

if __name__=='__main__':main()
