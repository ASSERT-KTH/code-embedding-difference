import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from jepa.tasks.decoder.train_lora import parse_args, train


if __name__ == "__main__":
    train(parse_args())
