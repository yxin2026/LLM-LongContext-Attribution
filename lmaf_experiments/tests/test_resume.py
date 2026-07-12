from pathlib import Path
from tempfile import TemporaryDirectory

from lmaf.utils.io import load_success_ids, write_jsonl


def test_terminal_skips_are_resume_completed() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.jsonl"
        write_jsonl(
            path,
            [
                {"sample_id": "ok", "error": None},
                {"sample_id": "skip", "error": "skipped_by_model_length"},
                {"sample_id": "retry", "error": "request_error"},
            ],
        )
        assert load_success_ids(path) == {"ok", "skip"}
