"""LLM reasoning layer: explains why a retrieved item matched, flags low-confidence/confusable
matches, and asks a clarifying question when species are genuinely hard to tell apart.

Uses the OpenAI API by default. To switch to Claude instead, replace the body of `_call_llm`
with an `anthropic.Anthropic().messages.create(...)` call — the rest of this class (prompt
construction, caching, confidence thresholding) is provider-agnostic.
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .cache import DiskCache

load_dotenv()  # reads OPENAI_API_KEY from a .env file in the working directory, if present

DEFAULT_MODEL = "gpt-4o-mini"

# Below this cosine similarity, treat a match as low-confidence regardless of what the LLM says.
CONFIDENCE_SCORE_THRESHOLD = 0.35

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species": {"type": "string"},
                    "explanation": {"type": "string"},
                    "low_confidence": {"type": "boolean"},
                },
                "required": ["species", "explanation", "low_confidence"],
            },
        },
        "clarifying_question": {"type": ["string", "null"]},
    },
    "required": ["explanations", "clarifying_question"],
}

SYSTEM_PROMPT = (
    "You are an ornithology assistant interpreting cross-modal bird retrieval results "
    "(audio call / photo / text description -> species matches). Given the query and top "
    "candidate matches with similarity scores, briefly explain why each plausible match fits, "
    "flag any match that looks unreliable, and — only if the top candidates are genuinely "
    "confusable species (e.g. from the provided confusable-pairs list) — ask ONE short "
    "clarifying question (e.g. about region or season) that would help disambiguate. If there's "
    "no real ambiguity, set clarifying_question to null.\n\n"
    "Respond ONLY with JSON matching this schema:\n" + json.dumps(RESPONSE_SCHEMA)
)


@dataclass
class ReasoningResult:
    explanations: list
    clarifying_question: str | None = None


class ReasoningLayer:
    def __init__(self, model: str = DEFAULT_MODEL, confusable_pairs: list | None = None,
                 cache: DiskCache | None = None):
        self.model = model
        self.confusable_pairs = confusable_pairs or []
        self.cache = cache or DiskCache()

    def explain(self, query_description: str, candidates: list[dict]) -> ReasoningResult:
        prompt = self._build_prompt(query_description, candidates)
        cached = self.cache.get(prompt)
        if cached is not None:
            return ReasoningResult(**cached)

        raw = self._call_llm(prompt)
        self.cache.set(prompt, raw)
        return ReasoningResult(**raw)

    def _build_prompt(self, query_description: str, candidates: list[dict]) -> str:
        candidate_species = {c["species"] for c in candidates}
        relevant_pairs = [
            p for p in self.confusable_pairs
            if {p["species_a"], p["species_b"]} & candidate_species
        ]
        payload = {
            "query": query_description,
            "candidates": [
                {
                    "species": c["species"],
                    "modality": c["modality"],
                    "similarity": round(c["score"], 3),
                    "low_confidence_by_score": c["score"] < CONFIDENCE_SCORE_THRESHOLD,
                }
                for c in candidates
            ],
            "known_confusable_pairs": relevant_pairs,
        }
        return json.dumps(payload, sort_keys=True)

    def _call_llm(self, prompt: str) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
