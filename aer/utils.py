""" Assorted utilities. """
import datetime

import pandas as pd

from asta import Array, typechecked


def parse_datetime(rep: str) -> str:
    """ Removes UTC offset. """
    segments = rep.split("+")
    if not segments:
        raise ValueError("Failed to parse datetime.")
    return segments[0]


@typechecked
def read_series(source_path: str) -> Array[float, -1]:
    """ Reads a CSV of MOER values into a single sequence. """
    # Assumes single-row header and format: ``<datetime>, <int>``.
    frame = pd.read_csv(source_path, sep=",")
    raw_series = frame.values

    # Convert strings to datetimes.
    for i, _ in enumerate(raw_series):
        stamp = parse_datetime(raw_series[i][0])
        raw_series[i][0] = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")

    series: Array[-1] = raw_series[:, 1]

    return series
