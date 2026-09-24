# Local RAG assistant

A small Python assistant that reads local documents, builds BM25 retrieval at runtime,
answers each question separately, and validates chunk citations before exporting results.
Retrieval and validation use only the Python standard library. The only dependencies are
`python-dotenv` and the OpenAI SDK. No API key is needed for the extractive fallback.

## Setup and run

Use Python 3.10 or newer. From the repository directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell or
`cp .env.example .env` in a Unix shell). Leave the keys empty for offline answers.

```bash
python run.py
python validate.py
python -m unittest discover -s tests -v
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe run.py` and
`.venv\Scripts\python.exe validate.py` directly. Input/output paths are relative to
the directory containing `pipeline.py`, regardless of the shell's working directory.
Both commands return nonzero on failures. Generated artifacts are included as an
example; deleting `artifacts/` and rerunning regenerates them entirely from disk.

### Optional LLM configuration

Set `LLM_PROVIDER=openai`, `LLM_MODEL` to a model available to your account that supports
Chat Completions JSON mode, and `OPENAI_API_KEY` in the environment or your private `.env`.
The example model is `gpt-4o-mini`; model access is account dependent.
For an OpenAI-compatible service, set `LLM_PROVIDER=openai_compatible`, `LLM_MODEL`,
`LLM_BASE_URL`, and `LLM_API_KEY`. Other native provider protocols are not implemented.
Existing environment variables take precedence over `.env`. Never commit `.env`.

For Groq, use the following `.env` settings and enter your key locally:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=
```

The existing SDK client works with [Groq's compatible endpoint](https://console.groq.com/docs/openai).
The [GPT OSS 20B model on Groq](https://console.groq.com/docs/model/openai/gpt-oss-20b)
supports JSON object mode. An empty key keeps the pipeline in offline fallback mode.

Each question with retrieved context gets its own API request containing only that
question and its labeled chunks, plus the fixed grounding instructions. JSON mode is
requested and the returned object is schema-checked. See the
[official JSON mode documentation](https://developers.openai.com/api/docs/guides/structured-outputs#json-mode).
API errors, invalid JSON, and wrong response schemas get one retry, then the exact
unsupported answer. SDK automatic retries are disabled. Each attempt has a 30-second
timeout and its own call-log record. No retrieved context means no API call.
Citation mistakes are preserved for validation; only duplicate citations are removed.
The full messages array is serialized as compact, sorted-key UTF-8 JSON for SHA-256 hashing.

## Stages

```text
INIT
 -> DOCUMENTS_LOADED
 -> CHUNKS_CREATED
 -> INDEX_BUILT
 -> QUESTIONS_LOADED
 -> RETRIEVAL_COMPLETE
 -> ANSWERS_GENERATED
 -> CITATIONS_VALIDATED
 -> RESULTS_EXPORTED
 -> VALIDATION_COMPLETE
```

Each stage is a separate function, prints its name when starting, and saves an artifact
before the next stage can start. `STAGE_ORDER` and `PipelineState` reject skipped,
repeated, or unfinished stages. The final stage imports `run_checks` from `validate.py`.

## Inputs, chunking, and retrieval

- All nonempty UTF-8 `.md`/`.txt` files under `docs/` are loaded recursively in sorted
  path order. A document ID is its relative path without its extension. Conflicting
  stems such as `guide.md` and `guide.txt` cause a clear error; rename one file.
- Chunks target 500 characters with up to 100 characters of overlap. In the latter
  half of a window, paragraph boundaries are preferred, then sentence boundaries,
  then whitespace. Overlap starts move forward to avoid cutting words where possible.
  A very long unbroken token can require a hard cut. Original whitespace and newline
  characters are preserved, and `text == document_text[start_char:end_char]`.
- IDs (`chunk_1`, etc.) follow sorted document IDs and character offsets. Identical
  inputs produce identical IDs. Editing the corpus can shift global IDs.
- BM25 uses lowercase regex tokens, a small stopword set, `k1=1.5`, and `b=0.75`.
  There is no stemming, embedding service, or precomputed index. All chunks are scored.
  Positive scores are sorted descending with lexical chunk-ID tie breaks. The top 3
  are divided by the question's maximum score and rounded to 4 decimal places.
- `questions.json` contains `{ "id": integer, "question": nonempty string }` records.
  Malformed entries are warned about and skipped. For duplicate IDs, the first valid
  entry wins and later entries are skipped. The validator applies this same policy:
  every accepted question must have exactly one answer. Invalid top-level JSON/schema
  fails the run. An empty question list is allowed. The supplied six questions are
  reproduced exactly; the five documents describe the fictional ForgeNest platform.

## Rule-based answers and grounding

With no provider API key, no network requests are made. The fallback considers only
the highest-ranking chunk and calculates the fraction of distinct question keywords
present in it. Below `FALLBACK_MIN_OVERLAP = 0.6`, it abstains. Otherwise it extracts
up to 3 sentences/lines with the most overlapping terms (excluding Markdown headings), restores their source order,
and cites that chunk. It never adds generated factual prose. `llm_calls.jsonl` stays
empty and the summary explicitly reports that no LLM calls occurred.

The exact abstention is:

> The knowledge base does not provide enough information to answer this.

It has `supported: false` and `citations: []`. The sample GraphQL question is unsupported.
The fallback is a lexical heuristic: paraphrases may be missed and overlapping terms
do not prove the extracted passage fully answers a question.

Validation checks that every citation belongs to that question's retrieval, supported
answers have citations, and unsupported answers have none. It also verifies numerical
claims against cited text and requires at least 50% of distinct answer keywords to
appear there. Numeric comparison preserves currency symbols, decimal precision, and
percent signs, normalizing thousands separators (`1,000` equals `1000`). Citation IDs
are excluded from numeric claims. The canonical abstention is exempt from grounding.
These checks detect some hallucinations; they do not prove semantic entailment or
that a number is attached to the correct claim. Invalid answers remain visible with
readable issues, and both commands fail rather than silently repairing their citations.

## Artifacts

| File under `artifacts/` | Contents |
| --- | --- |
| `run.json` | Mode, provider, model; no credentials |
| `documents.json` | Source IDs, relative paths, original text |
| `chunks.json` | Stable chunk IDs, document IDs, text, offsets |
| `index.json` | BM25 summary and parameters |
| `questions_loaded.json` | Accepted questions |
| `retrieval_results.json` | Per-question scored context |
| `answers.json` | Per-question answer, support flag, citations |
| `citation_validation.json` | Per-answer passed flag and issues |
| `final_answers.json` | Joined answers, retrieved IDs, validation |
| `llm_calls.jsonl` | Individual API attempts, or empty in fallback mode |
| `validation_report.json` | Final check results and summary |

The validator checks JSON syntax, schemas, source offsets, unique IDs, question coverage,
retrieval limits and source integrity, citation membership, grounding, final-output
consistency, and LLM log fields/counts. It reads the persisted run mode rather than your
current environment. One initial call per retrieved question is required in LLM mode;
a second record is allowed for its retry. Tests mock the SDK and use isolated temporary
fixtures to exercise failures without real API calls or credentials.
