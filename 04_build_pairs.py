from pathlib import Path
from pyspark.sql import SparkSession, Window, functions as F

project_dir = Path(__file__).resolve().parent

spark = (
    SparkSession.builder.master("local[2]")
    .appName("BuildHeadlinePairs")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

arms = spark.read.parquet(
    str(project_dir / "data/interim/comparable_arms")
)

# Give the same table two names so we can compare its rows.
a = arms.alias("a")
b = arms.alias("b")

# Match different headlines within the same comparable group.
match = (
    (F.col("a.clickability_test_id") == F.col("b.clickability_test_id"))
    & (F.col("a.eyecatcher_id") == F.col("b.eyecatcher_id"))
    & (F.col("a.excerpt") == F.col("b.excerpt"))
    & (F.col("a.headline") < F.col("b.headline"))
)

candidates = a.join(b, match, "inner").select(
    F.col("a.clickability_test_id").alias("test_id"),
    F.col("a.eyecatcher_id").alias("image_id"),
    F.col("a.excerpt").alias("excerpt"),
    F.col("a.row_id").cast("long").alias("row_id_a"),
    F.col("b.row_id").cast("long").alias("row_id_b"),
    F.col("a.headline").alias("headline_a"),
    F.col("b.headline").alias("headline_b"),
    F.col("a.impressions").alias("impressions_a"),
    F.col("b.impressions").alias("impressions_b"),
    F.col("a.clicks").alias("clicks_a"),
    F.col("b.clicks").alias("clicks_b"),
)

# Create a reproducible ordering without using outcomes.
candidates = candidates.withColumn(
    "selection_hash",
    F.sha2(
        F.to_json(F.struct(
            "test_id", "image_id", "excerpt",
            "headline_a", "headline_b"
        )),
        256,
    ),
)

ordering = Window.partitionBy("test_id").orderBy(
    "selection_hash", "row_id_a", "row_id_b"
)

# Keep exactly one pair per experiment.
pairs = (
    candidates
    .withColumn("pair_rank", F.row_number().over(ordering))
    .filter(F.col("pair_rank") == 1)
    .drop("pair_rank")
    .withColumn("ctr_a", F.col("clicks_a") / F.col("impressions_a"))
    .withColumn("ctr_b", F.col("clicks_b") / F.col("impressions_b"))
    .withColumn("lift_pp", 100 * (F.col("ctr_b") - F.col("ctr_a")))
    .cache()
)

pairs.write.mode("overwrite").parquet(
    str(project_dir / "data/interim/pairs")
)

print("Pair rows:", pairs.count())
print("Experiments:", pairs.select("test_id").distinct().count())

spark.stop()
