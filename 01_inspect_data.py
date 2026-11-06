from pathlib import Path
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("InspectUpworthy")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

path = Path(__file__).resolve().parent / "data/raw/upworthy.csv"

df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .option("mode", "FAILFAST")
    .csv(str(path))
    .cache()
)

print("Rows:", df.count())
print("Experiments:", df.select("clickability_test_id").distinct().count())
print("Columns:", df.columns)

df.select(
    "clickability_test_id", "headline", "impressions", "clicks"
).show(3, truncate=60)

spark.stop()
