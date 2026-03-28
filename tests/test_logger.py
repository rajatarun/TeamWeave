import importlib
import io
import logging
import os
import unittest


class LoggerTests(unittest.TestCase):
    def test_formatter_includes_extra_fields(self):
        logger_module = importlib.import_module("src.orchestrator.logger")
        logger_module = importlib.reload(logger_module)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logger_module._ExtraAwareFormatter(
                "%(asctime)sZ %(levelname)s %(name)s %(message)s",
                "%Y-%m-%dT%H:%M:%S",
            )
        )

        logger = logging.getLogger("extra_test")
        logger.handlers = []
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)

        logger.info("owner_profile_context_retrieval", extra={"top_k": 9, "owner": "Jane Doe"})
        output = stream.getvalue()

        self.assertIn('"top_k": 9', output)
        self.assertIn('"owner": "Jane Doe"', output)

    def test_get_logger_respects_log_level_for_existing_logger(self):
        logger_module = importlib.import_module("src.orchestrator.logger")
        logger_module = importlib.reload(logger_module)

        logger = logging.getLogger("level_update_test")
        logger.handlers = []
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(logging.StreamHandler(io.StringIO()))

        self.addCleanup(lambda: os.environ.pop("LOG_LEVEL", None))
        os.environ["LOG_LEVEL"] = "DEBUG"

        configured_logger = logger_module.get_logger("level_update_test")

        self.assertEqual(configured_logger.level, logging.DEBUG)
        self.assertTrue(all(handler.level == logging.DEBUG for handler in configured_logger.handlers))


if __name__ == "__main__":
    unittest.main()
