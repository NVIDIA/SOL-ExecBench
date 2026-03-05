from pathlib import Path
from typing import List

import pytest


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Skip tests marked requires_torch_cuda when CUDA is not available."""
    if _torch_cuda_available():
        return

    skip_cuda = pytest.mark.skip(reason="CUDA not available, skipping test")
    for item in items:
        if any(item.iter_markers(name="requires_torch_cuda")):
            item.add_marker(skip_cuda)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated temporary cache directory for each test.

    Sets SOLEXECBENCH_CACHE_PATH so every builder writes build artifacts into a
    fresh temp directory, preventing pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
    return cache_dir
