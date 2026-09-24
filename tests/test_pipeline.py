"""Regression tests for grounding, replacement inputs, stage order, and API failure paths."""

import contextlib
import io
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pipeline
from rag_utils import UNSUPPORTED_ANSWER, citation_issues, parse_json, read_json, valid_questions
from validate import run_checks


class PipelineTests(unittest.TestCase):
    """Use independent disk fixtures and mocked API responses; never access the network."""

    def setUp(self):
        """Create an isolated workspace and force fallback mode unless a test opts into mocks."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "", "LLM_API_KEY": "",
                                                  "LLM_PROVIDER": "openai", "LLM_MODEL": "test-model",
                                                  "LLM_BASE_URL": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def fixture(self):
        """Write renamed nested documents and nonsequential reordered question IDs."""
        (self.root / "docs/nested").mkdir(parents=True)
        (self.root / "docs/nested/renamed.txt").write_text(
            "Webhook delivery logs are retained for 17 days. "
            "An export costs $9 and includes 1000 entries.\n", encoding="utf-8")
        (self.root / "docs/empty.md").write_text("  \n", encoding="utf-8")
        self.write("questions.json", [{"id": 91, "question": "How long are webhook delivery logs retained?"},
                                      {"id": -3, "question": "GraphQL subscriptions?"}])

    def write(self, name, value):
        """Write an artifact or input fixture as JSON."""
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def run_pipeline(self):
        """Run quietly so unittest output is limited to useful results."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pipeline.main(self.root)

    def test_replacement_inputs_and_determinism(self):
        """Renamed files, changed counts/IDs, empty files and zero-match questions work."""
        self.fixture()
        self.assertEqual(self.run_pipeline(), 0)
        final = read_json(self.root / "artifacts/final_answers.json")
        self.assertEqual([r["id"] for r in final], [91, -3])
        self.assertIn("17 days", final[0]["answer"])
        self.assertTrue(final[0]["supported"])
        self.assertEqual(final[1]["answer"], UNSUPPORTED_ANSWER)
        self.assertFalse(final[1]["supported"])
        self.assertEqual((self.root / "artifacts/llm_calls.jsonl").read_text(), "")
        snapshots = {p.name: p.read_bytes() for p in (self.root / "artifacts").glob("*.json")}
        self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual(snapshots, {p.name: p.read_bytes() for p in (self.root / "artifacts").glob("*.json")})

    def test_stage_order_and_persistence(self):
        """Neither stage skipping nor advancing before output persistence is allowed."""
        state = pipeline.PipelineState(self.root)
        with self.assertRaises(RuntimeError):
            state.start("CHUNKS_CREATED")
        with contextlib.redirect_stdout(io.StringIO()):
            state.start("INIT")
        with self.assertRaises(RuntimeError):
            state.start("DOCUMENTS_LOADED")

    def test_chunk_offsets_overlap_and_long_tokens(self):
        """Long prose and unbroken tokens produce bounded chunks covering every character."""
        for text in ["A sentence about records.\r\n\r\n" * 100, "x" * 1600]:
            spans = list(pipeline.split_document(text))
            self.assertGreater(len(spans), 1)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], len(text))
            for start, end in spans:
                self.assertGreater(end, start)
                self.assertLessEqual(end - start, pipeline.CHUNK_SIZE)
            for previous, current in zip(spans, spans[1:]):
                self.assertGreater(current[0], previous[0])
                self.assertLessEqual(current[0], previous[1])
                self.assertLessEqual(previous[1] - current[0], pipeline.CHUNK_OVERLAP)

    def test_question_schema(self):
        """Skip booleans, missing fields, blank strings and duplicate IDs consistently."""
        raw = [{"id": 8, "question": "valid"}, {"id": 8, "question": "duplicate"},
               {"id": True, "question": "boolean"}, {"id": 7, "question": " "}, None]
        self.assertEqual(valid_questions(raw, warn=False), [{"id": 8, "question": "valid"}])
        with self.assertRaises(ValueError):
            valid_questions({})

    def test_bm25_ties_top_k_and_no_matches(self):
        """Equal scores use lexical chunk IDs, top-k is bounded, and zero matches are omitted."""
        self.fixture()
        state = pipeline.PipelineState(self.root)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.initialize(state)
            documents = pipeline.load_documents(state)
            pipeline.create_chunks(state, documents)
            chunks = [{"chunk_id": f"chunk_{i}", "doc_id": "same", "text": "alpha beta"}
                      for i in [2, 11, 1, 3]]
            index = pipeline.build_index(state, chunks)
            pipeline.load_questions(state)
            results = pipeline.retrieve(state, [{"id": 42, "question": "alpha"},
                                                {"id": 99, "question": "gamma"}], chunks, index)
        self.assertEqual([c["chunk_id"] for c in results[0]["retrieved_chunks"]],
                         ["chunk_1", "chunk_11", "chunk_2"])
        self.assertEqual([c["score"] for c in results[0]["retrieved_chunks"]], [1.0] * 3)
        self.assertEqual(results[1]["retrieved_chunks"], [])

    def test_missing_docs_and_colliding_stems_fail(self):
        """Missing sources and extension-free identity collisions fail clearly."""
        self.assertEqual(self.run_pipeline(), 1)
        self.fixture()
        (self.root / "docs/nested/renamed.md").write_text("Other content", encoding="utf-8")
        self.assertEqual(self.run_pipeline(), 1)

    def test_numeric_and_keyword_grounding(self):
        """Detect unseen numbers, substring tricks, unrelated prose and invalid citations."""
        retrieved = [{"chunk_id": "chunk_1", "text": "Logs last 30 days. Export costs $49 for 1000 records; uptime is 99.9%."}]
        answer = {"answer": "Logs last 30 days.", "supported": True, "citations": ["chunk_1"]}
        self.assertEqual(citation_issues(answer, retrieved), [])
        for prose in ["Logs last 3 days.", "Logs last 31.", "Export costs $9.", "Uptime is 99.8%.",
                      "Butterflies deliver extraordinary magic."]:
            self.assertTrue(citation_issues({**answer, "answer": prose}, retrieved), prose)
        self.assertTrue(citation_issues({**answer, "citations": ["chunk_2"]}, retrieved))
        self.assertTrue(citation_issues({**answer, "citations": []}, retrieved))
        self.assertEqual(citation_issues(pipeline.unsupported_answer(), []), [])

    def test_strict_model_schema_preserves_bad_citations(self):
        """Malformed schemas fail, while unknown citations remain available for validation."""
        for content in ['[]', '{"answer":"x","supported":"true","citations":[]}', '```json\n{}\n```']:
            with self.assertRaises(ValueError):
                pipeline.parse_llm_answer(content)
        result = pipeline.parse_llm_answer('{"answer":"x","supported":true,"citations":["wrong","wrong"]}')
        self.assertEqual(result["citations"], ["wrong"])
        with self.assertRaises(ValueError):
            parse_json('{"x": NaN}')

    def response(self, content):
        """Construct the SDK response shape without importing or invoking its network client."""
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def test_llm_retry_logs_and_question_isolation(self):
        """Invalid JSON retries once; zero-context questions never enter a model prompt."""
        self.fixture()
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        client = Mock()
        client.chat.completions.create.side_effect = [self.response("not JSON"), self.response(
            '{"answer":"Webhook delivery logs are retained for 17 days.","supported":true,"citations":["chunk_1"]}')]
        with patch("pipeline.make_client", return_value=client):
            self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        logs = [json.loads(line) for line in (self.root / "artifacts/llm_calls.jsonl").read_text().splitlines()]
        self.assertEqual([row["question_id"] for row in logs], [91, 91])
        self.assertEqual([row["attempt"] for row in logs], [1, 2])
        for call in client.chat.completions.create.call_args_list:
            self.assertNotIn("GraphQL", json.dumps(call.kwargs["messages"]))
        self.assertTrue(all(c["passed"] for c in run_checks(self.root, verbose=False)))

    def test_api_failure_abstains_after_two_attempts(self):
        """Two API errors yield the canonical unsupported answer and two real attempt logs."""
        self.fixture()
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("simulated API failure")
        with patch("pipeline.make_client", return_value=client):
            self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertTrue(all(not a["supported"] for a in read_json(self.root / "artifacts/answers.json")))

    def test_multiple_questions_are_never_batched(self):
        """Each retrieved question has its own request, prompt payload and log entry."""
        self.fixture()
        questions = [{"id": 25, "question": "How long are logs retained?"},
                     {"id": 4, "question": "What does an export cost?"}]
        self.write("questions.json", questions)
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        client = Mock()
        client.chat.completions.create.side_effect = [
            self.response('{"answer":"Logs are retained for 17 days.","supported":true,"citations":["chunk_1"]}'),
            self.response('{"answer":"An export costs $9.","supported":true,"citations":["chunk_1"]}'),
        ]
        with patch("pipeline.make_client", return_value=client):
            self.assertEqual(self.run_pipeline(), 0)
        prompts = [json.loads(c.kwargs["messages"][1]["content"])
                   for c in client.chat.completions.create.call_args_list]
        self.assertEqual([p["question"] for p in prompts], [q["question"] for q in questions])
        logs = [json.loads(line) for line in (self.root / "artifacts/llm_calls.jsonl").read_text().splitlines()]
        self.assertEqual([row["question_id"] for row in logs], [25, 4])

    def test_tampered_artifacts_fail(self):
        """The validator catches a forged passed flag and changed numeric claims."""
        self.fixture()
        self.assertEqual(self.run_pipeline(), 0)
        answers = read_json(self.root / "artifacts/answers.json")
        answers[0]["answer"] = "Webhook delivery logs are retained for 999 days."
        self.write("artifacts/answers.json", answers)
        checks = run_checks(self.root, verbose=False)
        self.assertFalse(all(c["passed"] for c in checks))
        self.assertTrue(any(not c["passed"] and "grounding" in c["name"] for c in checks))
        (self.root / "artifacts/answers.json").write_text("{broken", encoding="utf-8")
        self.assertFalse(all(c["passed"] for c in run_checks(self.root, verbose=False)))

    def test_empty_questions_and_tokenless_corpus(self):
        """Zero-question and zero-vocabulary corpora do not divide by zero or crash."""
        self.fixture()
        (self.root / "docs/nested/renamed.txt").write_text("--- !!!", encoding="utf-8")
        self.assertEqual(self.run_pipeline(), 0)
        self.assertTrue(all(not a["supported"] for a in read_json(self.root / "artifacts/answers.json")))
        self.write("questions.json", [])
        self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual(read_json(self.root / "artifacts/final_answers.json"), [])


if __name__ == "__main__":
    unittest.main()
