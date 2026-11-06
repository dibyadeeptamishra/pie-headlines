from pathlib import Path
from pyspark.sql import SparkSession, functions as F

project_dir = Path(__file__).resolve().parent
spark = (
    SparkSession.builder.master("local[2]")
    .appName("ComparableHeadlines")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

arms = spark.read.parquet(str(project_dir / "data/interim/arms"))

def tidy(column):
    text = F.coalesce(F.col(column), F.lit(""))
    return F.trim(F.regexp_replace(text, r"\s+", " "))

usable = (
    arms.withColumn("headline", tidy("headline"))
    .withColumn("excerpt", tidy("excerpt"))
    .withColumn("eyecatcher_id", tidy("eyecatcher_id"))
    .filter(
        (F.col("headline") != "") & (F.col("eyecatcher_id") != "")
    )
)

group_columns = ["clickability_test_id", "eyecatcher_id", "excerpt"]

groups = (
    usable.groupBy(*group_columns)
    .agg(F.count_distinct("headline").alias("n_headlines"))
    .filter(F.col("n_headlines") >= 2)
)

eligible = usable.join(
    groups.select(*group_columns), on=group_columns, how="inner"
).cache()

eligible.write.mode("overwrite").parquet(
    str(project_dir / "data/interim/comparable_arms")
)

print("Comparable groups:", groups.count())
print("Eligible experiments:", eligible.select("clickability_test_id").distinct().count())
print("Eligible variant rows:", eligible.count())

spark.stop()
