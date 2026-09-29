"""Candidate link selection. The navigator makes the real decision; this only caps the choice set."""

from __future__ import annotations

import re

from .wikipedia import Link

STOP = {
    "the", "and", "from", "with", "list", "lists", "county", "counties", "state", "states",
    "united", "national", "history", "born", "created", "town", "city", "area", "people",
    "who", "was", "were", "for", "near",
}


def destination_keywords(destination: str, summary: str = "") -> list[str]:
    """Distinctive words from the destination title and summary: proper nouns and long words."""
    keywords: set[str] = set()
    for phrase in re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", f"{destination}\n{summary}"):
        lowered = phrase.lower()
        if len(lowered) > 2 and lowered not in STOP:
            keywords.add(lowered)
        for word in lowered.split():
            if len(word) > 3 and word not in STOP:
                keywords.add(word)
    for word in re.split(r"\W+", f"{destination} {summary}".lower()):
        if len(word) > 4 and word not in STOP:
            keywords.add(word)
    return sorted(keywords)


def hop_score(link: Link, keywords: list[str]) -> int:
    haystack = f"{link.name} {link.title}".lower()
    score = 0
    for keyword in keywords:
        if keyword and keyword in haystack:
            score += 5 if len(keyword) > 6 else 3
    return score


def order_candidates(links: list[Link], keywords: list[str], max_links: int = 28) -> list[Link]:
    """Keyword matches first (best score first), then the rest in page order, capped."""
    scored = [(link, hop_score(link, keywords)) for link in links]
    matched = [link for link, score in sorted(
        (item for item in scored if item[1] > 0), key=lambda item: item[1], reverse=True
    )]
    rest = [link for link, score in scored if score == 0]
    return (matched + rest)[:max_links]
