"""Load native PostgreSQL plans and pg_stats for QueryFormer."""

import csv
from datetime import date, datetime
import json
import random
import re
import zlib

import numpy as np
import pandas as pd


def template_family(key):
    name = key.split(":", 1)[-1]
    name = re.sub(r"#\d+$", "", name)
    return re.sub(r"_s\d+$", "", name)


def load_plan_frames(path, split_seed=2027):
    records = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            if entry.get("kind") == "plan":
                records.append(entry["record"])
    families = sorted({template_family(record["key"]) for record in records})
    random.Random(split_seed).shuffle(families)
    train_families = set(families[:max(1, int(len(families) * 0.8))])

    def frame(selected):
        return pd.DataFrame({
            "id": range(len(selected)),
            "key": [record["key"] for record in selected],
            "json": [json.dumps(record["plan"]) for record in selected],
            "label_duration_ms": [record["label_duration_ms"] for record in selected],
        })

    train = [record for record in records
             if template_family(record["key"]) in train_families]
    test = [record for record in records
            if template_family(record["key"]) not in train_families]
    if not train or not test:
        raise ValueError("at least two template families are required")
    return frame(train), frame(test)


def _value(text):
    text = text.strip().strip('"')
    try:
        return float(text)
    except ValueError:
        try:
            return float(datetime.fromisoformat(text).timestamp())
        except ValueError:
            try:
                return float(date.fromisoformat(text).toordinal())
            except ValueError:
                return zlib.crc32(text.encode()) / 0xffffffff


def _bounds(text, bin_number):
    if not text:
        return np.linspace(0, 1, bin_number + 1).tolist()
    values = [_value(value) for value in next(csv.reader([text[1:-1]], escapechar='\\'))]
    if not values:
        return np.linspace(0, 1, bin_number + 1).tolist()
    values.sort()
    positions = np.linspace(0, len(values) - 1, bin_number + 1)
    return [values[round(position)] for position in positions]


def load_pg_stats(path, bin_number=50):
    with open(path, encoding="utf-8") as source:
        statistics = json.load(source)
    rows = []
    minimums = {}
    columns = {"NA": 0}
    for statistic in statistics:
        column = statistic["tablename"] + "." + statistic["attname"]
        bins = _bounds(statistic.get("histogram_bounds"), bin_number)
        rows.append({"table_column": column, "bins": bins})
        minimums[column] = (bins[0], bins[-1])
        columns[column] = len(columns)
    return pd.DataFrame(rows), minimums, columns
