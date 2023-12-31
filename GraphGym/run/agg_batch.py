import argparse
import os
import sys

# Add upper directory to sys path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from graphgym.utils.agg_runs import agg_batch


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model')
    parser.add_argument('--dir', dest='dir', help='Dir for batch of results',
                        required=True, type=str)
    parser.add_argument('--metric', dest='metric',
                        help='metric to select best epoch', required=False,
                        type=str, default='auto')
    return parser.parse_args()


args = parse_args()
agg_batch(args.dir, args.metric)
