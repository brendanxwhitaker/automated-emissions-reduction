""" Generate synthetic training data. """
import argparse
from aer.generation import generate


def main() -> None:
    """ Parses arguments and calls ``generate()``. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path", type=str, required=True, help="Path to seed training data."
    )
    parser.add_argument(
        "--seq-len", type=int, default=12, help="Sequence length for LSTM."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for LSTM."
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="Hidden size for LSTM."
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Num layers for LSTM."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of training epochs."
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
