"""Market data access for the quant research application."""

import logging as _logging

# Library practice: without a configured handler (outside the CLI, e.g. in tests) log
# records are dropped rather than printed; the CLI adds the file handler (quant/logs.py).
_logging.getLogger("quant").addHandler(_logging.NullHandler())
