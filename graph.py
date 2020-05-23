""" Generate synthetic training data. """
import argparse
from aer.vis import visualize


def main() -> None:
    """ Parses arguments and calls ``generate()``. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path", type=str, required=True, help="Path to dataset."
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path for saved svg."
    )
    parser.add_argument(
        "--settings-path", type=str, required=True, help="Path to plotplotplot settings file."
    )
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
