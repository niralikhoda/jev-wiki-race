"""Wikipedia access for the race: article fetch, link extraction, summaries, random pairs.

Ported from the browser code in davext/classifier-wiki-race. All calls go to the
public MediaWiki API. No proxy is needed from Python.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from urllib.parse import quote, unquote

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("wiki_race.wikipedia")

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "typesafe-wiki-race/0.1 (local experiment; python httpx)"
MAX_RETRIES = 4
BACKOFF_BASE = 1.5
MIN_SPACING_S = 0.4

SKIP_HREF = re.compile(
    r"/wiki/(File|Image|Help|Wikipedia|Template|Talk|Category|Portal|Module|Special|"
    r"Draft|Book|MediaWiki|TimedText|Template_talk|User):|redlink=1|action=edit",
    re.IGNORECASE,
)
CHROME_SELECTORS = (
    ".navbox, .infobox, .sidebar, .reflist, .references, .metadata, .mw-editsection, "
    ".hatnote, .thumb, .gallery, table, sup.reference, style"
)


@dataclass(frozen=True)
class Link:
    """One clickable article link, with a short stable ref used as the answer key."""

    ref: str
    name: str
    title: str


@dataclass
class Article:
    title: str
    url: str
    links: list[Link]
    html: str = ""


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", unquote(str(title or "")).replace("_", " ")).strip()


def title_key(title: str) -> str:
    return normalize_title(title).lower()


def article_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(normalize_title(title).replace(' ', '_'))}"


def parse_title_input(raw: str) -> str:
    """Accept a plain title or any Wikipedia URL form and return the clean title."""
    text = str(raw or "").strip()
    if not text:
        return ""
    in_url = re.search(r"wikipedia\.org/wiki/([^?#]+)", text, re.IGNORECASE)
    if in_url:
        return normalize_title(in_url.group(1))
    bare = re.match(r"^/?wiki/([^?#]+)", text, re.IGNORECASE)
    if bare:
        return normalize_title(bare.group(1))
    return normalize_title(text)


class Wikipedia:
    """Thin MediaWiki client scoped to what the race needs."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout)
        self._last_request = 0.0

    def _get(self, **params: object) -> dict:
        """GET with polite spacing and backoff on 429, which Wikipedia returns under bursts."""
        params["format"] = "json"
        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            response = self._client.get(API, params=params)
            if response.status_code != 429 or attempt == MAX_RETRIES:
                response.raise_for_status()
                return response.json()
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else BACKOFF_BASE * (2 ** attempt)
            logger.warning("Wikipedia 429, retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_SPACING_S:
            time.sleep(MIN_SPACING_S - elapsed)
        self._last_request = time.monotonic()

    def get_article(self, title: str) -> Article:
        """Fetch a rendered article and extract its body links as valid hops."""
        data = self._get(action="parse", page=title, prop="text|displaytitle", redirects=1)
        if "error" in data:
            raise ValueError(data["error"].get("info", "Article not found"))

        parsed = data["parse"]
        resolved = normalize_title(parsed["title"])
        soup = BeautifulSoup(parsed["text"]["*"], "html.parser")
        root = soup.select_one(".mw-parser-output") or soup

        for element in root.select(CHROME_SELECTORS):
            element.decompose()

        links: list[Link] = []
        seen: set[str] = set()
        for anchor in root.select("a[href]"):
            href = anchor.get("href", "")
            href = re.sub(r"^\./", "/wiki/", href)
            if not href.startswith("/wiki/") or SKIP_HREF.search(href):
                continue
            target = normalize_title(href.removeprefix("/wiki/").split("#")[0])
            key = title_key(target)
            if not target or key in seen:
                continue
            name = re.sub(r"\s+", " ", anchor.get_text() or target).strip()
            if not name or len(name) > 90:
                continue
            seen.add(key)
            links.append(Link(ref=f"e{len(links) + 1}", name=name, title=target))

        return Article(title=resolved, url=article_url(resolved), links=links, html=str(root))

    def get_summary(self, title: str, max_chars: int = 320) -> str:
        """Short intro extract, used to ground the navigator on what the destination is."""
        data = self._get(
            action="query", prop="extracts", exintro=1, explaintext=1, redirects=1, titles=title
        )
        pages = data.get("query", {}).get("pages", {})
        first = next(iter(pages.values()), {})
        extract = re.sub(r"\s+", " ", first.get("extract", "")).strip()
        if len(extract) > max_chars:
            return extract[:max_chars].strip() + "..."
        return extract

    def get_backlink_count(self, title: str, limit: int = 60) -> int:
        data = self._get(
            action="query", list="backlinks", bltitle=title, blnamespace=0,
            blfilterredir="all", bllimit=limit,
        )
        return len(data.get("query", {}).get("backlinks", []))

    def get_random_titles(self, count: int = 2) -> list[str]:
        data = self._get(action="query", list="random", rnnamespace=0, rnlimit=count)
        return [item["title"] for item in data.get("query", {}).get("random", [])]

    def get_random_validated_article(
        self, role: str = "start", min_links: int = 15, min_backlinks: int | None = None,
        max_tries: int = 20,
    ) -> Article:
        """A random article that is well connected in both directions."""
        needed = min_backlinks if min_backlinks is not None else (60 if role == "dest" else 40)
        last_error: Exception | None = None
        for _ in range(max_tries):
            try:
                title = self.get_random_titles(1)[0]
                article = self.get_article(title)
                if len(article.links) < min_links:
                    continue
                if self.get_backlink_count(article.title, needed + 5) < needed:
                    continue
                return article
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not find a well-connected random article: {last_error}")

    def get_reachable_random_pair(
        self, min_hops: int = 3, max_hops: int = 6, min_backlinks: int = 50
    ) -> tuple[Article, Article]:
        """Random-walk from a hub so the returned pair is guaranteed to be connected."""
        start = self.get_random_validated_article(role="start")
        visited = {title_key(start.title)}
        current = start
        dest: Article | None = None
        target_hops = random.randint(min_hops, max_hops)

        for hop in range(1, max_hops + 1):
            options = [l for l in current.links if title_key(l.title) not in visited]
            nxt: Article | None = None
            for _ in range(6):
                if not options:
                    break
                pick = options.pop(random.randrange(len(options)))
                try:
                    article = self.get_article(pick.title)
                except Exception:
                    continue
                if len(article.links) >= 12 and title_key(article.title) not in visited:
                    nxt = article
                    break
            if nxt is None:
                break
            visited.add(title_key(nxt.title))
            current = nxt
            if hop >= target_hops and self.get_backlink_count(current.title, min_backlinks + 5) >= min_backlinks:
                dest = current
                break

        dest = dest or current
        if title_key(dest.title) == title_key(start.title):
            raise RuntimeError("Random walk did not move. Try again.")
        logger.info("Random pair: %s -> %s (walk length %d)", start.title, dest.title, len(visited) - 1)
        return start, dest
