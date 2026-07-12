import json
import zipfile

from lmaf.data.longbench import prepare_longbench, read_zip_task_records


def test_prepare_longbench_from_local_jsonl(tmp_path) -> None:
    data_dir = tmp_path / "LongBench" / "data"
    data_dir.mkdir(parents=True)
    row = {
        "input": "What is the answer?",
        "context": "The answer is alpha.",
        "answers": ["alpha"],
        "length": 12,
        "dataset": "narrativeqa",
        "language": "en",
        "all_classes": [],
        "_id": "sample-1",
    }
    (data_dir / "narrativeqa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    written = prepare_longbench(["narrativeqa"], tmp_path / "processed", repo_dir=tmp_path / "LongBench")

    assert len(written) == 1
    prepared = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines()]
    assert prepared[0]["sample_id"] == "longbench_narrativeqa_000000"
    assert prepared[0]["answers"] == ["alpha"]
    assert "The answer is alpha." in prepared[0]["prompt"]


def test_read_zip_task_records(tmp_path) -> None:
    zip_path = tmp_path / "data.zip"
    row = {
        "input": "Where is the key?",
        "context": "The key is in the archive.",
        "answers": ["archive"],
        "length": 10,
        "dataset": "hotpotqa",
        "language": "en",
        "all_classes": [],
        "_id": "sample-2",
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/hotpotqa.jsonl", json.dumps(row) + "\n")

    rows = list(read_zip_task_records(zip_path, "hotpotqa"))

    assert rows == [row]
