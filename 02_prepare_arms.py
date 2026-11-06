from pathlib import Path
from pyspark.sql import SparkSession, functions as F

project_dir = Path(__file__).resolve().parent
spark = (
    SparkSession.builder.master("local[2]")
    .appName("PrepareUpworthy").getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

raw = spark.read.options(
    header=True, multiLine=True, escape='"', mode="FAILFAST"
).csv(str(project_dir / "data/raw/upworthy.csv"))

arms = (
    raw.withColumnRenamed("_c0", "row_id")
    .withColumn("created_date", F.to_date(F.substring("created_at", 1, 10)))
    .withColumn("impressions", F.col("impressions").cast("long"))
    .withColumn("clicks", F.col("clicks").cast("long"))
)

assert arms.filter(F.col("created_date").isNull()).count() == 0, "Missing dates"
assert arms.filter(F.col("clickability_test_id").isNull()).count() == 0, "Missing test IDs"

excluded_tests = (
    arms.filter(
        F.col("created_date").between("2013-06-25", "2014-01-10")
    )
    .select("clickability_test_id").distinct()
)

clean = arms.join(
    excluded_tests, on="clickability_test_id", how="left_anti"
).cache()

invalid_counts = (
    F.col("impressions").isNull() | F.col("clicks").isNull()
    | (F.col("impressions") <= 0)
    | (F.col("clicks") < 0)
    | (F.col("clicks") > F.col("impressions"))
)
assert clean.filter(invalid_counts).count() == 0, "Invalid counts found"

output_path = project_dir / "data/interim/arms"
output_path.parent.mkdir(parents=True, exist_ok=True)
clean.write.mode("overwrite").parquet(str(output_path))

print("Excluded experiments:", excluded_tests.count())
print("Remaining rows:", clean.count())
print("Remaining experiments:", clean.select("clickability_test_id").distinct().count())
print("Missing image IDs:", clean.filter(F.col("eyecatcher_id").isNull()).count())

spark.stop()
