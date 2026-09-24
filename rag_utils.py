"""Shared deterministic text, input, and grounding checks (standard library only)."""

import json
import re
import warnings

UNSUPPORTED_ANSWER = "The knowledge base does not provide enough information to answer this."
GROUNDING_MIN_OVERLAP = 0.5
STOPWORDS = frozenset("""
a an the and or but if then than as at by for from in into of on to with without
is are was were be been being do does did can could would should will may must
i me my we our you your it its they their this that these those there here
what which who when where why how all any some only also have has had not no
""".split())
NUMBER_PATTERN = re.compile(r"(?<![\w.])[$€£]?[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)")


def tokenize(text):
    """Lowercase Unicode words/numbers and remove a small fixed stopword set."""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


def parse_json(text):
    """Parse strict JSON, rejecting nonstandard NaN and Infinity constants."""
    def reject_constant(value):
        """Reject Python's otherwise permissive non-JSON numeric constants."""
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def read_json(path):
    """Read UTF-8 JSON, accepting an optional byte-order mark."""
    return parse_json(path.read_text(encoding="utf-8-sig"))


def valid_questions(raw, warn=True):
    """Skip malformed/duplicate entries; preserve valid integer IDs and source order."""
    if not isinstance(raw, list):
        raise ValueError("questions.json must contain a list of question records.")
    result, seen = [], set()
    for position, item in enumerate(raw):
        valid = (isinstance(item, dict) and type(item.get("id")) is int
                 and isinstance(item.get("question"), str) and item["question"].strip())
        if not valid or item["id"] in seen:
            if warn:
                warnings.warn(f"Skipping malformed or duplicate question entry at position {position}.")
            continue
        seen.add(item["id"])
        result.append({"id": item["id"], "question": item["question"]})
    return result


def citation_issues(answer, retrieved):
    """Check citation membership, support invariants, numeric and keyword grounding."""
    issues = []
    sources = {chunk["chunk_id"]: chunk["text"] for chunk in retrieved}
    citations = answer["citations"]
    for chunk_id in citations:
        if chunk_id not in sources:
            issues.append(f"Citation {chunk_id} was not retrieved for this question.")
    if answer["supported"] and not citations:
        issues.append("Supported answer has no citations.")
    if not answer["supported"] and citations:
        issues.append("Unsupported answer has citations.")
    # The fixed abstention makes no source claim and needs no textual grounding.
    if (not answer["supported"] and answer["answer"] == UNSUPPORTED_ANSWER
            and not citations):
        return issues
    if not answer["supported"] and answer["answer"] != UNSUPPORTED_ANSWER:
        issues.append("Unsupported answer does not use the required abstention text.")
    source_text = "\n".join(sources[c] for c in citations if c in sources)
    # Citation labels are metadata, not numerical claims in the answer prose.
    prose = answer["answer"]
    for chunk_id in citations:
        prose = re.sub(r"\b" + re.escape(chunk_id) + r"\b", "", prose)
    source_numbers = {n.replace(",", "") for n in NUMBER_PATTERN.findall(source_text)}
    for number in sorted(set(NUMBER_PATTERN.findall(prose))):
        if number.replace(",", "") not in source_numbers:
            issues.append(f"Number {number} is absent from the cited text.")
    keywords = set(tokenize(prose))
    overlap = len(keywords & set(tokenize(source_text))) / len(keywords) if keywords else 0.0
    if overlap < GROUNDING_MIN_OVERLAP:
        issues.append(f"Answer keyword grounding {overlap:.2f} is below {GROUNDING_MIN_OVERLAP:.2f}.")
    return issues
