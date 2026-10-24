"""Run the original CAnYA shared-supervised pretraining stage."""
import argparse
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', type=Path, required=True)
    cli, rest = p.parse_known_args()
    from teacher.train import cross_validate
    from teacher.utils.runtime import build_inference_model
    cross_validate.build_model = lambda args, device: build_inference_model(args, device, cli.backbone)
    from teacher.train import canya
    sys.argv = [canya.__file__, *rest]
    canya.main()

if __name__ == '__main__':
    main()
