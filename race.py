"""Run the Wikipedia race through one or more decision APIs and compare them.

Examples:
    python race.py --start "Sulla" --dest "Julius Caesar" --navigator jev
    python race.py --random --navigator jev claude
    python race.py --start "Ahmedabad" --dest "Mahatma Gandhi" --navigator jev claude --claude-model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from env_loader import load_env_file
from wiki_race.navigators import ClaudeNavigator, JevNavigator, Navigator, OpenAINavigator
from wiki_race.runner import RaceResult, run_race
from wiki_race.wikipedia import Wikipedia, parse_title_input

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpx2", "typesafe_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("wiki_race")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_navigators(names: list[str], claude_model: str, claude_effort: str, gpt_model: str) -> list[Navigator]:
    navigators: list[Navigator] = []
    for name in names:
        if name == "jev":
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise SystemExit("TYPESAFE_API_KEY is not set. Add it to .env.")
            navigators.append(JevNavigator())
        elif name == "claude":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env.")
            navigators.append(ClaudeNavigator(model=claude_model, effort=claude_effort))
        elif name == "gpt":
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY is not set. Add it to .env.")
            navigators.append(OpenAINavigator(model=gpt_model))
        else:
            raise SystemExit(f"Unknown navigator: {name}")
    return navigators


def print_summary(results: list[RaceResult]) -> None:
    header = f"{'navigator':<10} {'model':<16} {'outcome':<10} {'hops':>4} {'calls':>5} {'api ms':>8} {'wall ms':>8} {'cost usd':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.navigator:<10} {r.model[:16]:<16} {r.outcome:<10} {len(r.path) - 1:>4} {r.decisions:>5} "
            f"{r.total_latency_ms:>8} {r.wall_ms:>8} {r.total_cost_usd:>10.5f}"
        )
    print()
    for r in results:
        print(f"{r.navigator}: " + " -> ".join(r.path))
        if r.error:
            print(f"{r.navigator} error: {r.error}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Wikipedia race across decision APIs")
    parser.add_argument("--start", help="Start article title or URL")
    parser.add_argument("--dest", help="Destination article title or URL")
    parser.add_argument("--random", action="store_true", help="Pick a connected random pair")
    parser.add_argument("--navigator", nargs="+", default=["jev"], choices=["jev", "claude", "gpt"])
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--claude-model", default="claude-opus-5")
    parser.add_argument("--claude-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--gpt-model", default="gpt-4.1-mini")
    parser.add_argument("--no-save", action="store_true", help="Do not write a JSON record to results/")
    args = parser.parse_args()

    load_env_file()
    wiki = Wikipedia()

    if args.random:
        start_article, dest_article = wiki.get_reachable_random_pair()
        start, dest = start_article.title, dest_article.title
    elif args.start and args.dest:
        start, dest = parse_title_input(args.start), parse_title_input(args.dest)
    else:
        parser.error("Provide --start and --dest, or --random")
        return 2

    logger.info("Race: %s -> %s | navigators: %s", start, dest, ", ".join(args.navigator))
    navigators = build_navigators(args.navigator, args.claude_model, args.claude_effort, args.gpt_model)

    results = [run_race(wiki, nav, start, dest, max_steps=args.max_steps) for nav in navigators]
    print_summary(results)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS_DIR / f"{stamp}_{'-'.join(args.navigator)}.json"
        out.write_text(json.dumps([r.to_dict() for r in results], indent=2))
        logger.info("Saved %s", out)

    return 0 if all(r.outcome == "won" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
