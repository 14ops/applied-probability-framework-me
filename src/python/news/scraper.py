from __future__ import annotations

from typing import List, Tuple
import datetime as dt
import feedparser


def yahoo_finance_rss(symbol: str) -> str:
    sym = symbol.replace("/", "-").replace("^", "")
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"


def fetch_headlines(symbol: str, max_items: int = 20) -> List[Tuple[str, str, str]]:
    """
    Returns list of (title, link, published) tuples.
    """
    url = yahoo_finance_rss(symbol)
    feed = feedparser.parse(url)
    out: List[Tuple[str, str, str]] = []
    for e in feed.entries[:max_items]:
        title = getattr(e, "title", "")
        link = getattr(e, "link", "")
        published = getattr(e, "published", "")
        out.append((title, link, published))
    return out


