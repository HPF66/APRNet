"""Delegate to the original final-teacher trainer with rotary-key compatibility."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':

    import teacher.train.cross_validate as cross_validate
    from teacher.utils.compatibility import install_transfer_compat
    install_transfer_compat(cross_validate)
    from teacher.train.full_data import main
    main()
