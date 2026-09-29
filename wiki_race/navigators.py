"""Decision backends for one hop of the race.

Every navigator sees the same state and the same instructions and returns the same
Decision shape, so the race loop and the comparison report do not care which API
made the call. Differences in how confidence is obtained are deliberate and are the
point of the comparison:

- JevNavigator asks two Choice questions and reads probabilities the model emits.
- ClaudeNavigator asks for a JSON answer and a self-reported confidence number.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol

import anthropic
from pydantic import BaseModel, Field
from typesafe_sdk import Choice, ChoiceAnswer, TypeSafeClient

from .candidates import destination_keywords, order_candidates
from .prompts import HOP_ACTION, HOP_TARGET, OPERATIONS, build_state, build_target_criteria, shared_instructions
from .wikipedia import Article, Link

logger = logging.getLogger("wiki_race.navigators")

BRIDGE_THRESHOLD = 0.15
MAX_CANDIDATES = 28


@dataclass
class HopContext:
    goal: str
    destination: str
    destination_summary: str
    current: Article
    links: list[Link]
    visited: list[str]

    def candidates(self) -> list[Link]:
        keywords = destination_keywords(self.destination, self.destination_summary)
        return order_candidates(self.links, keywords, MAX_CANDIDATES)


@dataclass
class Decision:
    operation: str
    chosen: Link | None
    best_target: Link | None
    best_target_prob: float
    operation_probabilities: dict[str, float]
    operation_confidence: float | None
    target_probabilities: dict[str, float]
    target_confidence: float | None
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    candidates: list[Link] = field(default_factory=list)
    self_reported_confidence: float | None = None


class Navigator(Protocol):
    name: str

    def decide(self, ctx: HopContext) -> Decision: ...


def _rank_targets(probabilities: dict[str, float], candidates: list[Link]) -> tuple[Link | None, float]:
    ranked = sorted(
        ((ref, prob) for ref, prob in probabilities.items() if ref != "none"),
        key=lambda item: item[1], reverse=True,
    )
    if not ranked:
        return None, 0.0
    best_ref, best_prob = ranked[0]
    return next((l for l in candidates if l.ref == best_ref), None), best_prob


class JevNavigator:
    """Two Choice questions per hop, evaluated in parallel by Jev."""

    name = "jev"
    PRICE_INPUT_PER_M = 0.042

    def __init__(self, model: str | None = None, client: TypeSafeClient | None = None) -> None:
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.client = client or TypeSafeClient()

    def decide(self, ctx: HopContext) -> Decision:
        candidates = ctx.candidates()
        shared = shared_instructions(ctx.goal, ctx.destination, ctx.destination_summary, ctx.current.title)
        questions = {
            "operation": Choice(
                instructions={**shared, "rules": HOP_ACTION},
                criteria=OPERATIONS,
            ),
            "click_target": Choice(
                instructions={**shared, "operation": "CLICK", "rules": [HOP_ACTION, HOP_TARGET]},
                criteria=build_target_criteria(candidates),
            ),
        }
        state = build_state(ctx.current, ctx.destination, ctx.destination_summary, ctx.visited, candidates)

        started = perf_counter()
        response = self.client.system_one(state=state, questions=questions, model=self.model)
        latency_ms = round((perf_counter() - started) * 1000)

        operation = response.answers["operation"]
        target = response.answers["click_target"]
        if not isinstance(operation, ChoiceAnswer) or not isinstance(target, ChoiceAnswer):
            raise TypeError("Expected ChoiceAnswer for both questions")

        ref = None if target.choice == "none" else target.choice
        chosen = next((l for l in candidates if l.ref == ref), None)
        best_target, best_prob = _rank_targets(target.probabilities, candidates)
        input_tokens = response.usage.input_tokens or 0

        return Decision(
            operation=operation.choice,
            chosen=chosen,
            best_target=best_target,
            best_target_prob=best_prob,
            operation_probabilities=dict(operation.probabilities),
            operation_confidence=operation.confidence,
            target_probabilities=dict(target.probabilities),
            target_confidence=target.confidence,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=response.usage.output_tokens or 0,
            latency_ms=latency_ms,
            cost_usd=input_tokens * self.PRICE_INPUT_PER_M / 1_000_000,
            candidates=candidates,
        )


class HopAnswer(BaseModel):
    """What Claude returns per hop. Confidence here is self-reported, not measured."""

    operation: Literal["CLICK", "DONE", "BLOCKED"]
    click_target: str = Field(description='An offered element index such as "e3", or "none".')
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed certainty that click_target is the best move.")


CLAUDE_SYSTEM = (
    "You are the decision engine in a Wikipedia race. You receive the current page, the "
    "destination, the links on offer, and rules. Answer only with the JSON object requested. "
    "Do not explain."
)


class ClaudeNavigator:
    """One structured-output call per hop. Same state, same rules, JSON answer."""

    name = "claude"
    PRICES_PER_M: dict[str, tuple[float, float]] = {
        "claude-opus-5": (5.0, 25.0),
        "claude-sonnet-5": (2.0, 10.0),
        "claude-haiku-4-5": (1.0, 5.0),
    }

    def __init__(
        self, model: str = "claude-opus-5", effort: str = "low",
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def _request_kwargs(self) -> dict:
        # Haiku 4.5 supports neither the effort parameter nor adaptive thinking.
        if "haiku" in self.model:
            return {}
        return {"output_config": {"effort": self.effort}, "thinking": {"type": "adaptive"}}

    def decide(self, ctx: HopContext) -> Decision:
        candidates = ctx.candidates()
        shared = shared_instructions(ctx.goal, ctx.destination, ctx.destination_summary, ctx.current.title)
        payload = {
            "state": build_state(ctx.current, ctx.destination, ctx.destination_summary, ctx.visited, candidates),
            "questions": {
                "operation": {"instructions": {**shared, "rules": HOP_ACTION}, "criteria": OPERATIONS},
                "click_target": {
                    "instructions": {**shared, "operation": "CLICK", "rules": [HOP_ACTION, HOP_TARGET]},
                    "criteria": build_target_criteria(candidates),
                },
            },
            "answer_format": "Return operation, click_target (an element index or none) and confidence 0-1.",
        }

        started = perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload)}],
            output_format=HopAnswer,
            **self._request_kwargs(),
        )
        latency_ms = round((perf_counter() - started) * 1000)

        answer = response.parsed_output
        if answer is None:
            raise RuntimeError(f"Claude returned no parsable answer (stop_reason={response.stop_reason})")

        ref = None if answer.click_target == "none" else answer.click_target
        chosen = next((l for l in candidates if l.ref == ref), None)
        target_probabilities = {answer.click_target: answer.confidence}
        best_target, best_prob = _rank_targets(target_probabilities, candidates)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        price_in, price_out = self.PRICES_PER_M.get(self.model, (0.0, 0.0))

        return Decision(
            operation=answer.operation,
            chosen=chosen,
            best_target=best_target,
            best_target_prob=best_prob,
            operation_probabilities={answer.operation: answer.confidence},
            operation_confidence=answer.confidence,
            target_probabilities=target_probabilities,
            target_confidence=answer.confidence,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=(input_tokens * price_in + output_tokens * price_out) / 1_000_000,
            candidates=candidates,
        )


class OpenAINavigator:
    """One JSON-schema chat completion per hop, with token logprobs when the model allows.

    Two confidence numbers come back. The self-reported one is what the model wrote.
    The measured one is exp(sum of logprobs) over the tokens that spell click_target,
    which is the probability the model assigned to that exact answer string.
    """

    name = "gpt"
    PRICES_PER_M: dict[str, tuple[float, float]] = {
        "gpt-4.1": (2.0, 8.0),
        "gpt-4.1-mini": (0.4, 1.6),
        "gpt-4.1-nano": (0.1, 0.4),
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-5": (1.25, 10.0),
        "gpt-5-mini": (0.25, 2.0),
        "gpt-5-nano": (0.05, 0.4),
    }
    SCHEMA = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["CLICK", "DONE", "BLOCKED"]},
            "click_target": {"type": "string", "description": 'An offered element index such as "e3", or "none".'},
            "confidence": {"type": "number", "description": "Self-assessed certainty 0-1 that click_target is the best move."},
        },
        "required": ["operation", "click_target", "confidence"],
        "additionalProperties": False,
    }

    def __init__(self, model: str = "gpt-4.1-mini", client: "openai.OpenAI | None" = None, logprobs: bool = True) -> None:
        import openai

        self.model = model
        self.client = client or openai.OpenAI()
        self.logprobs = logprobs

    @staticmethod
    def _measured_probability(content: str, target: str, tokens: list) -> float | None:
        """Probability of the click_target string, from the logprobs of the tokens spanning it."""
        if not tokens or not target:
            return None
        marker = f'"click_target":'
        start = content.find(marker)
        if start < 0:
            return None
        value_start = content.find(target, start)
        if value_start < 0:
            return None
        value_end = value_start + len(target)
        pos = 0
        total = 0.0
        covered = False
        for tok in tokens:
            tok_start, tok_end = pos, pos + len(tok.token)
            pos = tok_end
            if tok_end <= value_start or tok_start >= value_end:
                continue
            total += tok.logprob
            covered = True
        return math.exp(total) if covered else None

    def decide(self, ctx: HopContext) -> Decision:
        candidates = ctx.candidates()
        shared = shared_instructions(ctx.goal, ctx.destination, ctx.destination_summary, ctx.current.title)
        payload = {
            "state": build_state(ctx.current, ctx.destination, ctx.destination_summary, ctx.visited, candidates),
            "questions": {
                "operation": {"instructions": {**shared, "rules": HOP_ACTION}, "criteria": OPERATIONS},
                "click_target": {
                    "instructions": {**shared, "operation": "CLICK", "rules": [HOP_ACTION, HOP_TARGET]},
                    "criteria": build_target_criteria(candidates),
                },
            },
            "answer_format": "Return operation, click_target (an element index or none) and confidence 0-1.",
        }
        kwargs: dict = {}
        if self.logprobs:
            kwargs.update(logprobs=True, top_logprobs=5)

        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLAUDE_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "hop_answer", "strict": True, "schema": self.SCHEMA}},
            max_completion_tokens=200,
            **kwargs,
        )
        latency_ms = round((perf_counter() - started) * 1000)

        choice = response.choices[0]
        content = choice.message.content or ""
        answer = HopAnswer.model_validate_json(content)
        tokens = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
        measured = self._measured_probability(content, answer.click_target, tokens)

        ref = None if answer.click_target == "none" else answer.click_target
        chosen = next((l for l in candidates if l.ref == ref), None)
        confidence = measured if measured is not None else answer.confidence
        target_probabilities = {answer.click_target: confidence}
        best_target, best_prob = _rank_targets(target_probabilities, candidates)

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        price_in, price_out = self.PRICES_PER_M.get(self.model, (0.0, 0.0))

        return Decision(
            operation=answer.operation,
            chosen=chosen,
            best_target=best_target,
            best_target_prob=best_prob,
            operation_probabilities={answer.operation: answer.confidence},
            operation_confidence=answer.confidence,
            target_probabilities=target_probabilities,
            target_confidence=confidence,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=(input_tokens * price_in + output_tokens * price_out) / 1_000_000,
            candidates=candidates,
            self_reported_confidence=answer.confidence,
        )
