import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def set_test_settings():
    from src.agent.graph import settings
    original_width = settings.graph_react_width
    settings.graph_react_width = 1
    yield
    settings.graph_react_width = original_width
