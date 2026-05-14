import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from jepa.tasks.decoder.test_projector import main


if __name__ == "__main__":
    main()
