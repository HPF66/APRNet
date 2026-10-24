"""Call the copied teacher CV implementation with a relocated saved config."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    cli = p.parse_args()

    import teacher.train.cross_validate as cross_validate
    from teacher.utils.compatibility import install_transfer_compat
    install_transfer_compat(cross_validate)
    from teacher.train.full_data import load_training_args
    from teacher.train.cross_validate import main as original_main
    original_main(load_training_args(cli.config))


if __name__ == '__main__':
    main()
