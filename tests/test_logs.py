import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.logs import LOGGER_NAME, close_logging, configure_logging, log_path
from tests.backtest_fixture import POOL, build_market_db, small_settings
from tests.settings_helper import SETTINGS

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


class ConfigureTests(unittest.TestCase):
    def test_the_log_sits_next_to_the_database_and_is_released_after_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "data" / "quant.db"
            handler = configure_logging(SETTINGS.logging, database)
            logging.getLogger("quant.test").warning("hello %s", "log")
            close_logging(handler)
            path = log_path(database)
            self.assertEqual(path, database.resolve().parent / "logs" / "quant.log")
            line = path.read_text(encoding="utf-8").strip()
            self.assertRegex(line, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ WARNING quant.test: hello log$")
        # leaving the with-block deleted the folder: the file was not held open

    def test_configuring_twice_keeps_one_file_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            first = configure_logging(SETTINGS.logging, Path(directory) / "a.db")
            second = configure_logging(SETTINGS.logging, Path(directory) / "a.db")
            files = [h for h in logging.getLogger(LOGGER_NAME).handlers if getattr(h, "_quant_file", False)]
            self.assertEqual(files, [second])
            self.assertIsNot(first, second)
            close_logging(second)


class CommandLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        build_market_db(self.root / "cli.db", small_settings())
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{(self.root / 'cli.db').as_posix()}"
        document["market"]["symbols"] = {sector: list(stocks) for sector, stocks in POOL.items()}
        document["positions"]["path"] = (self.root / "positions.csv").as_posix()
        self.config = self.root / "settings.yaml"
        self.config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.log = self.root / "logs" / "quant.log"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return cli.main(["--config", str(self.config), *args])

    def test_every_command_logs_its_start_and_finish(self):
        self.assertEqual(self.run_cli("status"), 0)
        text = self.log.read_text(encoding="utf-8")
        self.assertIn("INFO    quant.cli: command started: --config", text)
        self.assertRegex(text, r"command finished: status -> exit 0 in \d+\.\ds")

    def test_stages_and_their_warnings_are_logged(self):
        (self.root / "positions.csv").write_text("symbol,quantity\nTSLA,1\nCASH,1000\n", encoding="utf-8")
        self.assertEqual(self.run_cli("generate-signals"), 0)
        text = self.log.read_text(encoding="utf-8")
        self.assertIn("quant.pipeline: generate-signals/predictor SKIPPED", text)
        self.assertIn("quant.pipeline: generate-signals/signals OK (data to", text)
        self.assertIn("WARNING quant.signals.pipeline:", text)
        self.assertIn("TSLA is not in the stock pool", text)

    def test_refusals_are_errors_in_the_log(self):
        self.assertEqual(self.run_cli("promote", "nope"), 1)
        text = self.log.read_text(encoding="utf-8")
        self.assertIn("ERROR   quant.cli: Promotion refused: no model 'nope'", text)
        self.assertIn("WARNING quant.cli: command finished: promote -> exit 1", text)

    def test_a_crash_is_logged_with_its_traceback(self):
        def boom(args, settings):
            raise RuntimeError("disk on fire")

        with patch.dict(cli.COMMANDS, {"status": boom}), self.assertRaises(RuntimeError):
            self.run_cli("status")
        text = self.log.read_text(encoding="utf-8")
        self.assertIn("ERROR   quant.cli: command crashed:", text)
        self.assertIn("RuntimeError: disk on fire", text)
        self.assertIn("Traceback", text)


if __name__ == "__main__":
    unittest.main()
