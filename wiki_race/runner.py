"""The race loop, ported from the original app. The navigator decides; this only moves."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter

from .navigators import BRIDGE_THRESHOLD, Decision, HopContext, Navigator
from .prompts import build_state
from .wikipedia import Article, Wikipedia, title_key

logger = logging.getLogger("wiki_race.runner")


@dataclass
class HopRecord:
    step: int
    article: str
    operation: str
    chosen_title: str | None
    forced: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    operation_confidence: float | None
    target_confidence: float | None
    top_target_prob: float
    note: str = ""
    self_reported_confidence: float | None = None


@dataclass
class RaceResult:
    navigator: str
    model: str
    start: str
    destination: str
    outcome: str
    path: list[str]
    hops: list[HopRecord] = field(default_factory=list)
    wall_ms: int = 0
    error: str = ""

    @property
    def decisions(self) -> int:
        return len(self.hops)

    @property
    def total_latency_ms(self) -> int:
        return sum(h.latency_ms for h in self.hops)

    @property
    def total_cost_usd(self) -> float:
        return sum(h.cost_usd for h in self.hops)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            decisions=self.decisions,
            total_latency_ms=self.total_latency_ms,
            total_cost_usd=self.total_cost_usd,
        )
        return data


def _record(step: int, current: Article, decision: Decision, chosen_title: str | None, forced: bool, note: str = "") -> HopRecord:
    return HopRecord(
        step=step,
        article=current.title,
        operation=decision.operation,
        chosen_title=chosen_title,
        forced=forced,
        latency_ms=decision.latency_ms,
        input_tokens=decision.input_tokens,
        output_tokens=decision.output_tokens,
        cost_usd=decision.cost_usd,
        operation_confidence=decision.operation_confidence,
        target_confidence=decision.target_confidence,
        top_target_prob=decision.best_target_prob,
        note=note,
        self_reported_confidence=decision.self_reported_confidence,
    )


EventHook = Callable[[dict], None]


def run_race(
    wiki: Wikipedia, navigator: Navigator, start_title: str, dest_title: str, max_steps: int = 25,
    on_event: EventHook | None = None, step_delay_s: float = 0.0,
) -> RaceResult:
    """Race from start to destination. Returns a full record whatever the outcome.

    on_event, when given, receives a dict per notable step so a UI can follow along.
    """

    def emit(kind: str, **data: object) -> None:
        if on_event is not None:
            on_event({"kind": kind, **data})
    goal = f"find {dest_title} starting with {start_title}"
    result = RaceResult(
        navigator=navigator.name, model=getattr(navigator, "model", ""),
        start=start_title, destination=dest_title, outcome="running", path=[],
    )
    t0 = perf_counter()

    try:
        dest_key = title_key(dest_title)
        destination_summary = ""
        try:
            dest_article = wiki.get_article(dest_title)
            dest_key = title_key(dest_article.title)
            destination_summary = wiki.get_summary(dest_title)
        except Exception as exc:
            logger.warning("Could not preload destination %s: %s", dest_title, exc)

        current = wiki.get_article(start_title)
        visited: set[str] = {title_key(current.title)}
        tried: set[str] = set()
        trail: list[Article] = [current]
        result.path.append(current.title)
        logger.info("[%s] Loaded %s (%d links)", navigator.name, current.title, len(current.links))
        emit("loaded", article=current.title, links=len(current.links), destination_summary=destination_summary)

        if title_key(current.title) == dest_key:
            result.outcome = "won"
            return result

        def backtrack() -> bool:
            nonlocal current
            trail.pop()
            if not trail:
                return False
            current = trail[-1]
            logger.info("[%s] Backtracked to %s", navigator.name, current.title)
            emit("backtrack", article=current.title)
            return True

        for step in range(1, max_steps + 1):
            remaining = [l for l in current.links if title_key(l.title) not in visited and title_key(l.title) not in tried]
            if not remaining:
                if not backtrack():
                    result.outcome = "stuck"
                    return result
                continue

            ctx = HopContext(
                goal=goal, destination=dest_title, destination_summary=destination_summary,
                current=current, links=remaining, visited=sorted(visited),
            )
            decision = navigator.decide(ctx)
            emit(
                "decision", step=step, article=current.title, operation=decision.operation,
                chosen=decision.chosen.title if decision.chosen else None,
                latency_ms=decision.latency_ms, cost_usd=decision.cost_usd,
                input_tokens=decision.input_tokens, output_tokens=decision.output_tokens,
                operation_confidence=decision.operation_confidence,
                target_confidence=decision.target_confidence,
                self_reported_confidence=decision.self_reported_confidence,
                model=decision.model,
                operation_probabilities=decision.operation_probabilities,
                target_probabilities=decision.target_probabilities,
                candidates=[{"ref": l.ref, "label": l.name, "title": l.title} for l in decision.candidates],
                state=build_state(current, dest_title, destination_summary, sorted(visited), decision.candidates),
                top_targets=[
                    {"title": next((l.title for l in decision.candidates if l.ref == ref), ref), "prob": prob}
                    for ref, prob in sorted(decision.target_probabilities.items(), key=lambda kv: kv[1], reverse=True)[:5]
                ],
            )
            if step_delay_s > 0:
                time.sleep(step_delay_s)

            if decision.operation == "DONE" and title_key(current.title) == dest_key:
                result.hops.append(_record(step, current, decision, None, False, "done"))
                result.outcome = "won"
                return result

            chosen = decision.chosen
            forced = False
            if chosen is None and decision.best_target is not None and decision.best_target_prob >= BRIDGE_THRESHOLD:
                chosen = decision.best_target
                forced = True

            if chosen is None:
                result.hops.append(_record(step, current, decision, None, False, "no lead"))
                logger.info("[%s] %s: %s, no lead (%d ms)", navigator.name, current.title, decision.operation, decision.latency_ms)
                if not backtrack():
                    result.outcome = "stuck"
                    return result
                continue

            tried.add(title_key(chosen.title))
            logger.info(
                "[%s] %s -> %s%s | op=%s conf=%s | %d ms | $%.6f",
                navigator.name, current.title, chosen.title, " (bridge)" if forced else "",
                decision.operation,
                "n/a" if decision.target_confidence is None else f"{decision.target_confidence:.2f}",
                decision.latency_ms, decision.cost_usd,
            )

            try:
                nxt = wiki.get_article(chosen.title)
            except Exception as exc:
                result.hops.append(_record(step, current, decision, chosen.title, forced, f"skipped: {exc}"))
                emit("skipped", article=chosen.title, reason=str(exc)[:120])
                continue

            next_key = title_key(nxt.title)
            tried.add(next_key)

            if next_key == dest_key:
                result.hops.append(_record(step, current, decision, chosen.title, forced, "arrived"))
                result.path.append(nxt.title)
                result.outcome = "won"
                emit("moved", article=nxt.title)
                return result

            if next_key in visited:
                result.hops.append(_record(step, current, decision, chosen.title, forced, "loops back"))
                emit("loop", article=nxt.title)
                continue

            result.hops.append(_record(step, current, decision, chosen.title, forced))
            visited.add(next_key)
            current = nxt
            trail.append(nxt)
            result.path.append(nxt.title)
            emit("moved", article=nxt.title)

        result.outcome = "max_steps"
        return result
    except Exception as exc:
        logger.error("[%s] Race failed: %s", navigator.name, exc)
        result.outcome = "error"
        result.error = str(exc)
        return result
    finally:
        result.wall_ms = round((perf_counter() - t0) * 1000)
        emit("finished", outcome=result.outcome, path=result.path, error=result.error, summary=result.to_dict())
