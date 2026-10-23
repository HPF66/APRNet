"""Build aligned backbone graphs, then structure context graphs."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--backbone-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['backbone', 'context'], required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--structure-root', type=Path, action='append', default=[], help='Repeatable directory of original coordinates (or its parent)')
    p.add_argument('--dssp', default=None, help='DSSP/mkdssp executable; otherwise discover it on PATH')
    cli = p.parse_args()
    if cli.workers < 1:
        p.error('--workers must be positive')
    if cli.stage == 'backbone':
        from protein.scratch.get_feature import backbone_graphs as engine
        engine.DATA = cli.source.resolve()
        engine.CACHE = cli.backbone_output.resolve()
        engine.STRUCTURE_ROOTS = tuple(root.resolve() for root in cli.structure_root)
        engine.main(cli.workers)
    else:
        from protein.scratch.get_feature import context_graphs as engine
        engine.SOURCE = cli.source.resolve()
        engine.OLD_GRAPH = cli.backbone_output.resolve()
        engine.GRAPH = cli.output.resolve()
        engine.STRUCTURE_ROOTS = tuple(root.resolve() for root in cli.structure_root)
        engine.DSSP_EXECUTABLE = cli.dssp
        engine.main(cli.workers)

if __name__ == '__main__':
    main()
