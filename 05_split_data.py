from pathlib import Path
import hashlib
import re
from pyspark.sql import SparkSession

project_dir = Path(__file__).resolve().parent

spark = (
    SparkSession.builder.master("local[2]")
    .appName("SplitHeadlineExperiments")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

pairs = spark.read.parquet(str(project_dir / "data/interim/pairs"))
test_ids = sorted(
    row.test_id for row in pairs.select("test_id").distinct().collect()
)

# Read metadata from ALL cleaned variants of eligible experiments.
metadata = (
    spark.read.parquet(str(project_dir / "data/interim/arms"))
    .withColumnRenamed("clickability_test_id", "test_id")
    .join(pairs.select("test_id"), on="test_id", how="left_semi")
    .select("test_id", "slug", "headline")
    .collect()
)

def normalize(value):
    return re.sub(
        r"\s+", " ", value or "", flags=re.ASCII
    ).strip(" ").lower()

# Initially, each experiment belongs to its own group.
parent = {test_id: test_id for test_id in test_ids}

def find(test_id):
    while parent[test_id] != test_id:
        parent[test_id] = parent[parent[test_id]]
        test_id = parent[test_id]
    return test_id

# Merge groups whenever experiments share a headline or slug.
seen = {}
for row in metadata:
    for field in ("slug", "headline"):
        value = normalize(row[field])
        if not value:
            continue

        key = (field, value)
        if key in seen:
            a, b = find(row.test_id), find(seen[key])
            parent[max(a, b)] = min(a, b)
        else:
            seen[key] = row.test_id

# Assign whole groups using a fixed, reproducible hash.
assignments = []
for test_id in test_ids:
    group = find(test_id)
    digest = hashlib.sha256(("20261106:" + group).encode()).hexdigest()
    bucket = int(digest, 16) % 100

    if bucket < 60:
        split = "train"
    elif bucket < 80:
        split = "validation"
    else:
        split = "test"

    assignments.append((test_id, group, split))

split_map = spark.createDataFrame(
    assignments, ["test_id", "content_group", "split"]
)
dataset = pairs.join(split_map, on="test_id", how="inner").cache()

assert dataset.count() == len(test_ids), "Unexpected number of pairs"

output = project_dir / "data/processed/pairs"
output.parent.mkdir(parents=True, exist_ok=True)
dataset.write.mode("overwrite").parquet(str(output))

print("Related-content groups:", len({row[1] for row in assignments}))
dataset.groupBy("split").count().orderBy("split").show()

spark.stop()
