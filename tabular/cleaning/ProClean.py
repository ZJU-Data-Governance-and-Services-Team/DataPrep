import os

try:
    from .SemanticCleaner import run_semantic_cleaner
    from .project_config import (
        DEFAULT_FASTTEXT_MODEL_PATH,
        DEFAULT_LLM_API_KEY,
        DEFAULT_LLM_BASE_URL,
        DEFAULT_LLM_MODEL,
        DEFAULT_RESULT_ROOT,
        DEFAULT_SEMANTIC_MODEL_PATH,
    )
except ImportError:
    from SemanticCleaner import run_semantic_cleaner
    from project_config import (
        DEFAULT_FASTTEXT_MODEL_PATH,
        DEFAULT_LLM_API_KEY,
        DEFAULT_LLM_BASE_URL,
        DEFAULT_LLM_MODEL,
        DEFAULT_RESULT_ROOT,
        DEFAULT_SEMANTIC_MODEL_PATH,
    )


# Validate that a required path exists.
def _require_path(name: str, value: str):
    if not value:
        raise ValueError(f"{name} is required.")
    if not os.path.exists(value):
        raise FileNotFoundError(f"{name} does not exist: {value}")


# Validate that a required text value is present.
def _require_text(name: str, value: str):
    if not value:
        raise ValueError(f"{name} is required.")


class ProClean:
    """Run the full ProClean data cleaning pipeline."""

    def __init__(
        self,
        dataset: str,
        dirty_path: str,
        clean_path: str,
        result_root: str = DEFAULT_RESULT_ROOT,
        debug_mode: bool = False,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
        llm_model: str = DEFAULT_LLM_MODEL,
        semantic_model_path: str = DEFAULT_SEMANTIC_MODEL_PATH,
        fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH,
    ):
        self.dataset = dataset
        self.dirty_path = dirty_path
        self.clean_path = clean_path
        self.result_root = result_root
        self.debug_mode = debug_mode
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.semantic_model_path = semantic_model_path
        self.fasttext_model_path = fasttext_model_path

    def run(self) -> dict:
        """Run the full ProClean pipeline and return result metadata."""
        return run_proclean(
            dataset=self.dataset,
            dirty_path=self.dirty_path,
            clean_path=self.clean_path,
            result_root=self.result_root,
            debug_mode=self.debug_mode,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
            semantic_model_path=self.semantic_model_path,
            fasttext_model_path=self.fasttext_model_path,
        )


# Run the full ProClean pipeline and return result metadata.
def run_proclean(
    dataset: str,
    dirty_path: str,
    clean_path: str,
    result_root: str = DEFAULT_RESULT_ROOT,
    debug_mode: bool = False,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_api_key: str = DEFAULT_LLM_API_KEY,
    llm_model: str = DEFAULT_LLM_MODEL,
    semantic_model_path: str = DEFAULT_SEMANTIC_MODEL_PATH,
    fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH,
) -> dict:
    """Run the full ProClean pipeline from dirty data to final cleaned output.

    This is the only recommended project-level entrypoint. Every execution runs
    the complete pipeline: AutoLabeler -> FormatCleaner -> FDCleaner ->
    SemanticCleaner. Reusing previous phase outputs is not supported.

    Args:
        dataset: Dataset name used in output file names.
        dirty_path: Original dirty CSV path.
        clean_path: Clean CSV path used as ground truth.
        result_root: Root directory for all phase outputs.
        debug_mode: If True, semantic repair LLM calls return debug text.
        llm_base_url: Required OpenAI-compatible LLM base URL.
        llm_api_key: Required LLM API key.
        llm_model: Required LLM model name.
        semantic_model_path: Required local Qwen/small LM path used by
            SemanticDetector.
        fasttext_model_path: Required local FastText model path used for
            clustering.

    Returns:
        A dictionary containing result paths and runtime information.
    """
    _require_text("dataset", dataset)
    _require_path("dirty_path", dirty_path)
    _require_path("clean_path", clean_path)
    _require_path("semantic_model_path", semantic_model_path)
    _require_path("fasttext_model_path", fasttext_model_path)
    _require_text("llm_base_url", llm_base_url)
    _require_text("llm_api_key", llm_api_key)
    _require_text("llm_model", llm_model)

    result = run_semantic_cleaner(
        dataset=dataset,
        original_dirty_path=dirty_path,
        clean_path=clean_path,
        result_root=result_root,
        debug_mode=debug_mode,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        model_path=semantic_model_path,
        fasttext_model_path=fasttext_model_path,
    )

    result["config"] = {
        "dataset": dataset,
        "dirty_path": dirty_path,
        "clean_path": clean_path,
        "result_root": result_root,
        "debug_mode": debug_mode,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "semantic_model_path": semantic_model_path,
        "fasttext_model_path": fasttext_model_path,
    }
    return result


run_project = run_proclean


if __name__ == "__main__":
    result = run_proclean(
        dataset="beers",
        dirty_path="data/datasets/beers/beers_error-01.csv",
        clean_path="data/datasets/beers/beers_clean.csv",
        result_root=DEFAULT_RESULT_ROOT,
        debug_mode=False,
        llm_base_url=DEFAULT_LLM_BASE_URL,
        llm_api_key=DEFAULT_LLM_API_KEY,
        llm_model=DEFAULT_LLM_MODEL,
        semantic_model_path=DEFAULT_SEMANTIC_MODEL_PATH,
        fasttext_model_path=DEFAULT_FASTTEXT_MODEL_PATH,
    )
    print(result)
