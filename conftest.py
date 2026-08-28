# Repo-root collection safety (audit #31).
#
# `testpaths` scopes a bare `pytest` to sage/tests, but `pytest .` or `pytest sage/`
# escapes that and would import manual scripts scattered outside sage/tests that call
# sys.exit / load models at import time. Ignore those trees at collection so an
# accidental broad invocation cannot detonate them.

collect_ignore_glob = [
    "test_*.py",                 # repo-root manual scripts
    "sage/test_*.py",            # sage/-root manual scripts
    "sage/experiments/**/*.py",
    "sage/orchestration/test_*.py",
    "sage/quantization/*.py",
]
