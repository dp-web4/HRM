"""
HuggingFace Hub helpers for SAGE LoRA adapters.

Requires: pip install sage-cognition[hub]

Usage:
    from sage.hub import download_adapter, upload_adapter, list_adapters

    # Download a community adapter
    path = download_adapter("dp-web4/sage-lora-empathy-v1")

    # Upload your trained adapter
    upload_adapter("./checkpoints/my_lora", "myuser/sage-lora-custom")

    # List available adapters
    adapters = list_adapters()
"""

from pathlib import Path
from typing import Optional


def _ensure_hub():
    """Ensure huggingface_hub is installed."""
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for hub operations. "
            "Install with: pip install sage-cognition[hub]"
        )


def download_adapter(repo_id: str, revision: str = "main",
                     cache_dir: Optional[str] = None) -> Path:
    """
    Download a SAGE LoRA adapter from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "dp-web4/sage-lora-empathy-v1")
        revision: Git revision (branch, tag, or commit hash)
        cache_dir: Optional local cache directory

    Returns:
        Path to the downloaded adapter directory
    """
    hub = _ensure_hub()
    local_dir = hub.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
    )
    return Path(local_dir)


def upload_adapter(adapter_path: str | Path, repo_id: str,
                   commit_message: str = "Upload SAGE LoRA adapter",
                   private: bool = False) -> str:
    """
    Upload a LoRA adapter to HuggingFace Hub.

    Args:
        adapter_path: Local path to the adapter directory
        repo_id: Target HuggingFace repo ID
        commit_message: Commit message for the upload
        private: Whether to create a private repository

    Returns:
        URL of the uploaded repository
    """
    hub = _ensure_hub()
    api = hub.HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def list_adapters(author: str = "dp-web4") -> list:
    """
    List available SAGE adapters on HuggingFace.

    Args:
        author: HuggingFace author/org to search (default: dp-web4)

    Returns:
        List of dicts with adapter metadata (id, description, downloads)
    """
    hub = _ensure_hub()
    api = hub.HfApi()
    models = api.list_models(author=author, search="sage-lora")
    return [
        {
            "id": m.modelId,
            "downloads": getattr(m, "downloads", 0),
            "tags": getattr(m, "tags", []),
        }
        for m in models
    ]
