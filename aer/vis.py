""" Trains a time-series prediction model to generate synthetic training data. """
import argparse
import pandas as pd
from plotplotplot.draw import graph


def visualize(args: argparse.Namespace) -> None:
    """ Chart and save to file. """
    # Assumes single-row header and format: ``<datetime>, <int>``.
    frame = pd.read_csv(args.source_path, sep=",")
    frame = frame.drop(["timestamp"], axis=1)
    graph([frame], ["MOERS"], [1], args.save_path, args.settings_path)
    print(frame)
