# Databricks notebook source
# MAGIC %md
# MAGIC # Load kiip_sample_data.xlsx into tables
# MAGIC
# MAGIC Each sheet becomes a table with the same name in `<catalog>.<schema>`.
# MAGIC
# MAGIC Setup:
# MAGIC 1. Keep `kiip_sample_data.xlsx` in the same Git folder as this notebook. If it isn't there, upload it to the volume `main.kiip_sample.kiip_raw` instead.
# MAGIC 2. Run all cells. The notebook writes one Parquet file per sheet into the volume, then builds each table from its file.
# MAGIC
# MAGIC Re-running replaces the tables with the current file.

# COMMAND ----------

# MAGIC %pip install -q openpyxl

# COMMAND ----------

import datetime as dt
import os
import re
import sys

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kiip_sample")
dbutils.widgets.text("volume", "kiip_raw")
dbutils.widgets.text("excel_file", "kiip_sample_data.xlsx")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
excel_file = dbutils.widgets.get("excel_file")

volume_dir = f"/Volumes/{catalog}/{schema}/{volume}"
parquet_dir = f"{volume_dir}/parquet"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")
os.makedirs(parquet_dir, exist_ok=True)


def find_excel():
    """Prefer the workbook in the Git folder next to this notebook, then the one in the volume."""
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = context.notebookPath().get()
    if not notebook_path.startswith("/Workspace"):
        notebook_path = "/Workspace" + notebook_path
    candidates = [
        os.path.join(os.path.dirname(notebook_path), excel_file),
        f"{volume_dir}/{excel_file}",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"{excel_file} not found at {candidates[0]} or {candidates[1]}. "
        "Add it to the Git folder or upload it to the volume, then run again."
    )


excel_path = find_excel()
print(f"Source workbook: {excel_path}")

# COMMAND ----------

def infer_type(values):
    """Pick the Arrow type for one column from its non-empty values."""
    present = [v for v in values if v is not None]
    if not present:
        return pa.string()
    if all(isinstance(v, bool) for v in present):
        return pa.bool_()
    if all(isinstance(v, dt.datetime) for v in present):
        if all(v.time() == dt.time(0) for v in present):
            return pa.date32()
        return pa.timestamp("us")
    if all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return pa.int64()
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return pa.float64()
    return pa.string()


def to_arrow_value(value, arrow_type):
    if value is None:
        return None
    if pa.types.is_string(arrow_type):
        return str(value)
    if pa.types.is_date32(arrow_type):
        return value.date()
    return value


def sheet_to_table(ws):
    if not IDENTIFIER.match(ws.title):
        raise ValueError(f"Sheet name '{ws.title}' cannot be used as a table name")

    rows = ws.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        raise ValueError(f"Sheet '{ws.title}' is empty")

    names = [str(h).strip() if h is not None else "" for h in header]
    if any(not IDENTIFIER.match(n) for n in names):
        raise ValueError(f"Sheet '{ws.title}' has a blank or invalid column name: {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"Sheet '{ws.title}' has duplicate column names")

    # Empty Excel cells and blank rows are treated as NULL.
    data = []
    for row in rows:
        row = [None if v == "" else v for v in row] + [None] * (len(names) - len(row))
        if any(v is not None for v in row):
            data.append(row[: len(names)])

    arrays, fields = [], []
    for i, name in enumerate(names):
        values = [row[i] for row in data]
        arrow_type = infer_type(values)
        if pa.types.is_string(arrow_type) and any(v is not None and not isinstance(v, str) for v in values):
            print(f"  warning: {ws.title}.{name} mixes types; stored as text", file=sys.stderr)
        arrays.append(pa.array([to_arrow_value(v, arrow_type) for v in values], type=arrow_type))
        fields.append(pa.field(name, arrow_type))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))

# COMMAND ----------

# Write one typed Parquet file per sheet into the volume.
workbook = openpyxl.load_workbook(excel_path, data_only=True)
loaded = []
for ws in workbook.worksheets:
    print(f"Reading sheet '{ws.title}'")
    table = sheet_to_table(ws)
    path = f"{parquet_dir}/{ws.title}.parquet"
    pq.write_table(table, path)
    loaded.append((ws.title, path, table.num_rows))
    print(f"  {table.num_rows} rows, {table.num_columns} columns -> {path}")
workbook.close()

# COMMAND ----------

# Build each table from its Parquet file. Replaces the table if it already exists.
for table_name, path, _ in loaded:
    print(f"Creating table {catalog}.{schema}.{table_name}")
    spark.sql(
        f"CREATE OR REPLACE TABLE {catalog}.{schema}.{table_name} AS "
        f"SELECT * FROM read_files('{path}', format => 'parquet')"
    )

# COMMAND ----------

# Check every table's row count against its sheet.
mismatches = []
for table_name, _, expected in loaded:
    actual = spark.table(f"{catalog}.{schema}.{table_name}").count()
    status = "OK" if actual == expected else "MISMATCH"
    print(f"  {table_name}: table={actual} sheet={expected} {status}")
    if actual != expected:
        mismatches.append(table_name)

if mismatches:
    raise RuntimeError(f"Row count mismatch: {mismatches}")
print(f"Done. {len(loaded)} tables in {catalog}.{schema}")
