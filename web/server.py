"""Local server for the wiki race page.

Serves web/index.html and a small JSON API. Races run one at a time in a background
thread so Wikipedia never sees concurrent bursts. API keys stay in this process.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env_loader import load_env_file  # noqa: E402
from wiki_race.navigators import ClaudeNavigator, JevNavigator, Navigator, OpenAINavigator  # noqa: E402
from wiki_race.runner import run_race  # noqa: E402
from wiki_race.wikipedia import Article, Wikipedia, parse_title_input, title_key  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpx2", "typesafe_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("wiki_race.web")

INDEX_HTML = Path(__file__).resolve().parent / "index.html"
PORT = int(os.environ.get("PORT", "8765"))

LANES: dict[str, dict] = {
    "jev": {"label": "Jev", "models": ["jev-latest"]},
    "claude": {"label": "Claude", "models": ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]},
    "gpt": {"label": "GPT", "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-5-mini", "gpt-5"]},
}


def make_navigator(lane: str, model: str) -> Navigator:
    if lane == "jev":
        return JevNavigator(model=model)
    if lane == "gpt":
        # GPT-5 models refuse logprobs, so those fall back to self-reported confidence.
        return OpenAINavigator(model=model, logprobs=not model.startswith("gpt-5"))
    return ClaudeNavigator(model=model, effort="low")


class CachingWikipedia(Wikipedia):
    """Keeps the sanitized HTML of every article fetched so the page can show it."""

    def __init__(self) -> None:
        super().__init__()
        self.html_cache: dict[str, str] = {}

    def get_article(self, title: str) -> Article:
        article = super().get_article(title)
        self.html_cache[title_key(article.title)] = article.html
        return article


class RaceState:
    """Shared state for all lanes plus the single-race lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running: str | None = None
        self.wiki = CachingWikipedia()
        self.lanes: dict[str, dict] = {lane: self._empty(lane) for lane in LANES}

    def _empty(self, lane: str) -> dict:
        return {"lane": lane, **LANES[lane], "model": LANES[lane]["models"][0], "status": "idle", "events": [], "summary": None}

    def reset(self) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("A race is running")
            self.lanes = {lane: self._empty(lane) for lane in LANES}

    def snapshot(self) -> dict:
        with self.lock:
            return {"running": self.running, "lanes": self.lanes}

    def start(self, lane: str, start: str, dest: str, model: str | None, delay_s: float = 0.0) -> None:
        if lane not in LANES:
            raise ValueError(f"Unknown lane {lane}")
        model = model or LANES[lane]["models"][0]
        if model not in LANES[lane]["models"]:
            raise ValueError(f"Unknown model {model} for lane {lane}")
        with self.lock:
            if self.running:
                raise RuntimeError(f"{LANES[self.running]['label']} is still racing")
            self.running = lane
            self.lanes[lane] = {**self._empty(lane), "model": model, "status": "running"}
        threading.Thread(target=self._run, args=(lane, start, dest, model, delay_s), daemon=True).start()

    def _run(self, lane: str, start: str, dest: str, model: str, delay_s: float) -> None:
        def on_event(event: dict) -> None:
            with self.lock:
                self.lanes[lane]["events"].append(event)

        try:
            navigator = make_navigator(lane, model)
            result = run_race(self.wiki, navigator, start, dest, on_event=on_event, step_delay_s=delay_s)
            with self.lock:
                self.lanes[lane]["status"] = result.outcome
                self.lanes[lane]["summary"] = result.to_dict()
        except Exception as exc:
            logger.error("Lane %s failed: %s", lane, exc)
            with self.lock:
                self.lanes[lane]["status"] = "error"
                self.lanes[lane]["events"].append({"kind": "finished", "outcome": "error", "error": str(exc)})
        finally:
            with self.lock:
                self.running = None


STATE = RaceState()



class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug(fmt, *args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(HTTPStatus.OK, STATE.snapshot())
        elif self.path.startswith("/api/article"):
            title = parse_qs(urlparse(self.path).query).get("title", [""])[0]
            html = STATE.wiki.html_cache.get(title_key(title))
            if html is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "article not fetched yet"})
            else:
                self._json(HTTPStatus.OK, {"title": title, "html": html})
        elif self.path == "/api/keys":
            self._json(HTTPStatus.OK, {
                "jev": bool(os.environ.get("TYPESAFE_API_KEY")),
                "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "gpt": bool(os.environ.get("OPENAI_API_KEY")),
            })
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            if self.path == "/api/start":
                body = self._body()
                start = parse_title_input(body.get("start", ""))
                dest = parse_title_input(body.get("dest", ""))
                if not start or not dest:
                    raise ValueError("Start and destination are required")
                STATE.start(body.get("lane", ""), start, dest, body.get("model"), float(body.get("delay", 0) or 0))
                self._json(HTTPStatus.ACCEPTED, {"ok": True})
            elif self.path == "/api/random":
                start, dest = STATE.wiki.get_reachable_random_pair()
                self._json(HTTPStatus.OK, {"start": start.title, "dest": dest.title})
            elif self.path == "/api/reset":
                STATE.reset()
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except RuntimeError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except Exception as exc:
            logger.error("Request failed: %s", exc)
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main() -> int:
    load_env_file(ROOT / ".env")
    missing = [k for k in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        logger.warning("Missing keys, those lanes will fail: %s", ", ".join(missing))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    logger.info("Wiki race page on http://127.0.0.1:%d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
