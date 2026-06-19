import unittest
import tempfile
from pathlib import Path

from lba.logging_utils import create_run_logger, default_log_dir


class LoggingUtilsTest(unittest.TestCase):
    def test_default_log_dir(self) -> None:
        self.assertEqual(default_log_dir(Path("/tmp/project")), Path("/tmp/project/.lba/logs"))

    def test_create_run_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger, path = create_run_logger(tmpdir)
            logger.info("hello")

        self.assertTrue(path.name.startswith("lba-"))


if __name__ == "__main__":
    unittest.main()
