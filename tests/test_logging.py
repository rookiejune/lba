import tempfile
import unittest
from pathlib import Path

from lba.logging_utils import (
    JsonlEventWriter,
    create_run_logger,
    default_log_dir,
    event_log_path_for,
)


class LoggingUtilsTest(unittest.TestCase):
    def test_default_log_dir(self) -> None:
        self.assertEqual(default_log_dir(Path("/tmp/project")), Path("/tmp/project/.lba/logs"))

    def test_create_run_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger, path = create_run_logger(tmpdir)
            logger.info("hello")

        self.assertTrue(path.name.startswith("lba-"))
        self.assertEqual(event_log_path_for(path), path.with_suffix(".jsonl"))

    def test_jsonl_event_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            writer = JsonlEventWriter(path)
            writer.write("summary", {"padding": {"after": 0.1}})
            text = path.read_text()

        self.assertIn('"event": "summary"', text)
        self.assertIn('"padding": {"after": 0.1}', text)


if __name__ == "__main__":
    unittest.main()
