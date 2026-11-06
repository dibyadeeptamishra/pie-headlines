from pathlib import Path
import csv
from zipfile import ZipFile, ZIP_DEFLATED
from pyspark.sql import SparkSession

project_dir = Path(__file__).resolve().parent

spark = (
    SparkSession.builder.master("local[2]")
    .appName("ExportModelData")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

data = spark.read.parquet(
    str(project_dir / "data/processed/pairs")
)

output_dir = project_dir / "data/processed"
csv_paths = []

# Write one ordinary CSV file for each split.
for split in ("train", "validation", "test"):
    rows = (
        data.filter(data["split"] == split)
        .orderBy("test_id")
        .collect()
    )

    path = output_dir / f"{split}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data.columns)
        writer.writeheader()
        writer.writerows(row.asDict() for row in rows)

    csv_paths.append(path)
    print(f"{path.name}: {len(rows)} rows")

# Bundle the CSV files and preparation scripts.
bundle = project_dir / "pie-headlines-handoff.zip"
with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as archive:
    for path in csv_paths:
        archive.write(path, arcname=f"data/processed/{path.name}")

    for path in sorted(project_dir.glob("[0-9][0-9]_*.py")):
        archive.write(path, arcname=path.name)

print("Created:", bundle)
spark.stop()
