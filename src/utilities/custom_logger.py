import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pytz import timezone

TZ = timezone('Asia/Singapore')
CURRENT = datetime.now(tz=TZ)
TIME = CURRENT.strftime("%Y_%m_%d_%H_%M")
FORMAT = "[%(name)s: %(asctime)s, %(levelname)s] - %(message)s"

ROOT = Path(__file__).parents[0]
logging.basicConfig(filemode="a")


class CustomLogger(logging.Logger):
    """A customized logger class

    Raises:
        Exception: user must choose where to display the logged messages

    """
    def __init__(self,
                 name: str = __name__,
                 level: int = logging.INFO,
                 stream_console: Any = sys.stdout,
                 logger_path: Optional[Path] = None,
                 format: Optional[str] = FORMAT,
                 **kwargs):
        """create and configure logger
        """
        super().__init__(name, **kwargs)
        self.setLevel(level)
        formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S")
        if stream_console is None and logger_path is None:
            raise ValueError(
                "either stream_console or logger_path must be specified")

        if stream_console:
            # log at console
            console_handler = logging.StreamHandler(stream=stream_console)
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

        if logger_path:
            if not logger_path.parents[0].exists():
                logger_path.parents[0].mkdir(parents=True, exist_ok=True)

            # save log to a file
            file_handler = logging.FileHandler(f"{logger_path}")
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
