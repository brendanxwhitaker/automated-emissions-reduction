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
def read_series(source_path: str, start: str, end: str) -> Array[float, -1]:
    """ Reads a CSV of MOER values into a single sequence. """
    # Assumes single-row header and format: ``<datetime>, <int>``.
    frame = pd.read_csv(source_path, sep=",")
    raw_series = frame.values

    startdate = None
    enddate = None

    if start:
        startdate = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    if end:
        enddate = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    start_idx = 0
    end_idx = len(raw_series)

    # Convert strings to datetimes.
    for i, _ in enumerate(raw_series):
        stamp = parse_datetime(raw_series[i][0])
        date = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        raw_series[i][0] = date

        if startdate and date == startdate:
            print("Starting at:", startdate)
            start_idx = i

        if enddate and date == enddate + datetime.timedelta(hours=1):
            print("Ending at:", enddate)
            end_idx = i

    moers = raw_series[start_idx:end_idx, 1]
    series: Array[-1] = moers.astype(float)

    return series
