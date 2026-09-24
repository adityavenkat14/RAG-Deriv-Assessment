"""Standalone artifact validation, also imported by the pipeline's final stage."""

import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag_utils import citation_issues, parse_json, read_json, valid_questions

REQUIRED_FILES = [
    "questions.json", "artifacts/documents.json", "artifacts/chunks.json",
    "artifacts/index.json", "artifacts/questions_loaded.json", "artifacts/run.json",
    "artifacts/retrieval_results.json", "artifacts/answers.json",
    "artifacts/citation_validation.json", "artifacts/final_answers.json",
]
LOG_FIELDS = {"stage", "timestamp", "provider", "model", "question_id", "prompt_hash",
              "input_artifacts", "output_artifact"}


def require(condition, message):
    """Raise a readable validation failure, including under Python's optimized mode."""
    if not condition:
        raise ValueError(message)


def records(value, fields):
    """Validate a list of dictionaries and required typed fields, rejecting bool-as-int."""
    require(isinstance(value, list), "Expected a JSON list.")
    for item in value:
        require(isinstance(item, dict), "Expected an object record.")
        for name, kind in fields.items():
            require(name in item and type(item[name]) is kind, f"Missing or invalid field: {name}.")
    return value


def exact_coverage(items, question_ids):
    """Ensure exactly one record per accepted question, regardless of input ordering."""
    ids = [item["id"] for item in items]
    require(all(type(i) is int for i in ids), "Question IDs must be integers.")
    require(Counter(ids) == Counter(question_ids), "Question IDs are missing, duplicated, or unexpected.")


def run_checks(root=None, verbose=True):
    """Read current artifacts and return PASS/FAIL records; malformed files never crash the CLI."""
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    checks, data, log_rows = [], {}, []

    def check(name, operation):
        """Isolate one check so subsequent diagnostics still run after a failure."""
        try:
            operation()
            result = {"name": name, "passed": True, "issues": []}
        except Exception as exc:
            result = {"name": name, "passed": False, "issues": [str(exc)]}
        checks.append(result)
        if verbose:
            detail = ": " + "; ".join(result["issues"]) if result["issues"] else ""
            print(f"{'PASS' if result['passed'] else 'FAIL'} {name}{detail}")

    def files_exist():
        """Require the inputs and all persisted pipeline stage artifacts."""
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        require(not missing, "Missing files: " + ", ".join(missing))

    def parse_files():
        """Parse all input/artifact JSON and every log line, including blank-line errors."""
        names = set(REQUIRED_FILES)
        names.update(p.relative_to(root).as_posix() for p in (root / "artifacts").rglob("*.json"))
        errors = []
        for name in sorted(names):
            try:
                data[name] = read_json(root / name)
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: {exc}")
        log = root / "artifacts/llm_calls.jsonl"
        if log.exists():
            for number, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    row = parse_json(line)
                    require(isinstance(row, dict), "Expected a JSON object.")
                    log_rows.append(row)
                except ValueError as exc:
                    errors.append(f"llm_calls.jsonl line {number}: {exc}")
        require(not errors, "; ".join(errors))

    def documents_and_chunks():
        """Check source schemas, unique identities, references, and exact source offsets."""
        docs = records(data["artifacts/documents.json"], {"doc_id": str, "path": str, "text": str})
        chunks = records(data["artifacts/chunks.json"],
                         {"chunk_id": str, "doc_id": str, "text": str, "start_char": int, "end_char": int})
        require(bool(docs) and bool(chunks), "No documents or chunks.")
        doc_map = {d["doc_id"]: d for d in docs}
        require(len(doc_map) == len(docs), "Duplicate doc_id.")
        require(len({c["chunk_id"] for c in chunks}) == len(chunks), "Duplicate chunk_id.")
        for chunk in chunks:
            require(chunk["doc_id"] in doc_map, "Unknown source doc_id.")
            text = doc_map[chunk["doc_id"]]["text"]
            start, end = chunk["start_char"], chunk["end_char"]
            require(0 <= start < end <= len(text), "Invalid chunk offsets.")
            require(chunk["text"] == text[start:end], "Chunk text does not match source offsets.")

    def questions_and_answers():
        """Apply the shared input skip policy, then require one typed answer per accepted ID."""
        questions = valid_questions(data["questions.json"], warn=False)
        require(data["artifacts/questions_loaded.json"] == questions, "Loaded questions differ from disk inputs.")
        answers = records(data["artifacts/answers.json"],
                          {"id": int, "question": str, "answer": str, "supported": bool, "citations": list})
        exact_coverage(answers, [q["id"] for q in questions])
        questions_by_id = {q["id"]: q["question"] for q in questions}
        for answer in answers:
            require(answer["question"] == questions_by_id[answer["id"]], "Answer question text mismatch.")
            require(bool(answer["answer"].strip()), "Empty answer.")
            require(all(isinstance(c, str) for c in answer["citations"]), "Citation IDs must be strings.")
            require(len(set(answer["citations"])) == len(answer["citations"]), "Duplicate citations.")

    def retrieval_check():
        """Check top-three limits, finite scores, question coverage, and original source text."""
        retrieval = records(data["artifacts/retrieval_results.json"],
                            {"id": int, "question": str, "retrieved_chunks": list})
        questions = valid_questions(data["questions.json"], warn=False)
        exact_coverage(retrieval, [q["id"] for q in questions])
        question_map = {q["id"]: q["question"] for q in questions}
        chunks = {c["chunk_id"]: c for c in data["artifacts/chunks.json"]}
        for row in retrieval:
            require(row["question"] == question_map[row["id"]], "Retrieved question text mismatch.")
            selected = records(row["retrieved_chunks"], {"chunk_id": str, "doc_id": str, "text": str})
            require(len(selected) <= 3, "More than three retrieved chunks.")
            require(len({c["chunk_id"] for c in selected}) == len(selected), "Duplicate retrieved chunks.")
            for chunk in selected:
                source = chunks.get(chunk["chunk_id"])
                require(source is not None, "Unknown retrieved chunk.")
                require(all(source[k] == chunk[k] for k in ("doc_id", "text")), "Retrieved source text mismatch.")
                score = chunk.get("score")
                require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                        "Retrieval score must be finite and normalized to 0-1.")
            if selected:
                require(selected[0]["score"] == 1, "Best retrieved score must be 1.")
                require(all(a["score"] >= b["score"] for a, b in zip(selected, selected[1:])),
                        "Retrieved scores must be descending.")

    def citations_check():
        """Ensure valid source/retrieval citations and the supported/unsupported invariants."""
        retrieval = {r["id"]: r["retrieved_chunks"] for r in data["artifacts/retrieval_results.json"]}
        chunks = {c["chunk_id"] for c in data["artifacts/chunks.json"]}
        for answer in data["artifacts/answers.json"]:
            allowed = {c["chunk_id"] for c in retrieval[answer["id"]]}
            require(all(c in chunks and c in allowed for c in answer["citations"]), "Invalid citation ID.")
            require(bool(answer["citations"]) == answer["supported"], "Support/citation invariant failed.")

    def grounding_check():
        """Recompute grounding and compare it to saved results rather than trusting a passed flag."""
        validations = records(data["artifacts/citation_validation.json"], {"id": int, "passed": bool, "issues": list})
        questions = valid_questions(data["questions.json"], warn=False)
        exact_coverage(validations, [q["id"] for q in questions])
        saved = {v["id"]: v for v in validations}
        retrieval = {r["id"]: r["retrieved_chunks"] for r in data["artifacts/retrieval_results.json"]}
        errors = []
        for answer in data["artifacts/answers.json"]:
            issues = citation_issues(answer, retrieval[answer["id"]])
            require(saved[answer["id"]] == {"id": answer["id"], "passed": not issues, "issues": issues},
                    "Saved citation validation differs from recomputed checks.")
            errors.extend(f"Question {answer['id']}: {issue}" for issue in issues)
        require(not errors, "; ".join(errors))

    def final_check():
        """Require complete final records and consistency with the independently saved stages."""
        final = records(data["artifacts/final_answers.json"], {"id": int, "validation": dict, "retrieved_chunk_ids": list})
        questions = valid_questions(data["questions.json"], warn=False)
        exact_coverage(final, [q["id"] for q in questions])
        answers = {a["id"]: a for a in data["artifacts/answers.json"]}
        retrieval = {r["id"]: r for r in data["artifacts/retrieval_results.json"]}
        validations = {v["id"]: v for v in data["artifacts/citation_validation.json"]}
        for record in final:
            question_id = record["id"]
            require(all(record.get(k) == v for k, v in answers[question_id].items()), "Final answer differs from answers.json.")
            require(record["retrieved_chunk_ids"] == [c["chunk_id"] for c in retrieval[question_id]["retrieved_chunks"]],
                    "Final retrieved chunk IDs differ.")
            require(record["validation"] == {k: validations[question_id][k] for k in ("passed", "issues")},
                    "Final validation differs.")

    def logging_check():
        """Check saved run mode and one initial call plus at most one retry per retrieved question."""
        run = data["artifacts/run.json"]
        require(run.get("mode") in {"llm", "rule_based"}, "Missing/invalid run mode.")
        if run["mode"] == "rule_based":
            require(not log_rows, "Rule-based mode must not log fake LLM calls.")
            return
        require((root / "artifacts/llm_calls.jsonl").is_file(), "Missing LLM call log.")
        expected = {r["id"] for r in data["artifacts/retrieval_results.json"] if r["retrieved_chunks"]}
        counts = Counter()
        for row in log_rows:
            require(LOG_FIELDS <= row.keys(), "LLM call log is missing required fields.")
            require(row["stage"] == "answer_generation", "Unexpected LLM stage.")
            require(type(row["question_id"]) is int and row["question_id"] in expected, "Unexpected LLM question ID.")
            require(row["provider"] == run["provider"] and row["model"] == run["model"], "LLM configuration mismatch.")
            require(isinstance(row["prompt_hash"], str) and re.fullmatch(r"[0-9a-f]{64}", row["prompt_hash"]),
                    "Invalid SHA-256 prompt hash.")
            require(datetime.fromisoformat(row["timestamp"]).tzinfo is not None, "Timestamp needs an ISO-8601 timezone.")
            require(row["input_artifacts"] == ["artifacts/retrieval_results.json"]
                    and row["output_artifact"] == "artifacts/answers.json", "Incorrect log artifact paths.")
            counts[row["question_id"]] += 1
            require(row.get("attempt", counts[row["question_id"]]) == counts[row["question_id"]], "Invalid attempt order.")
        require(set(counts) == expected and all(1 <= count <= 2 for count in counts.values()),
                "Each retrieved question needs one call record, plus at most one retry record.")

    for name, operation in [
        ("Required files exist", files_exist), ("JSON and JSONL syntax", parse_files),
        ("Document/chunk schemas, IDs and offsets", documents_and_chunks),
        ("Exactly one answer per valid question", questions_and_answers),
        ("Retrieval coverage, limits and source integrity", retrieval_check),
        ("Citation IDs and support flags", citations_check),
        ("Citation validation coverage and recomputed grounding", grounding_check),
        ("Final answer coverage and consistency", final_check),
        ("LLM call logging matches run mode", logging_check),
    ]:
        check(name, operation)
    return checks


def main():
    """Print all checks and return a failing exit status if any check fails."""
    try:
        return 0 if all(c["passed"] for c in run_checks()) else 1
    except (OSError, ValueError) as exc:
        print(f"FAIL validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
