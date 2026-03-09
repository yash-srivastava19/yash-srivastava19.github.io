#!/usr/bin/env python3
"""Fetch Letterboxd RSS and write to _data/letterboxd.yml."""

import re
import os
import yaml
import feedparser
from datetime import datetime, timezone
from html.parser import HTMLParser


class _StripHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts).strip()


def strip_html(html: str) -> str:
    p = _StripHTML()
    p.feed(html)
    return p.get_text()


def numeric_to_stars(rating) -> str:
    if rating is None:
        return ""
    try:
        r = float(rating)
    except (ValueError, TypeError):
        return ""
    full = int(r)
    half = 1 if (r - full) >= 0.5 else 0
    return "★" * full + ("½" if half else "")


def extract_poster(html: str) -> str:
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html)
    return m.group(1) if m else ""


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_existing(data_path: str) -> list:
    if not os.path.exists(data_path):
        return []
    with open(data_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("films", [])


def parse_entry(entry) -> dict:
    ns = entry.get("tags", [])

    def get_ns(key):
        for t in entry.get("letterboxd_" + key, []):
            return t
        # feedparser flattens namespaced fields differently
        return entry.get("letterboxd_" + key) or entry.get(key)

    title = entry.get("letterboxd_filmtitle") or entry.get("title", "")
    year = entry.get("letterboxd_filmyear", "")
    rating_raw = entry.get("letterboxd_memberrating")
    watched_date = entry.get("letterboxd_watcheddate", "")
    rewatch_raw = entry.get("letterboxd_rewatch", "No")
    rewatch = str(rewatch_raw).strip().lower() in ("yes", "true", "1")
    link = entry.get("link", "")
    description_html = entry.get("description", "") or entry.get("summary", "")
    poster = extract_poster(description_html)
    review = strip_html(description_html)
    # Remove the film title line that Letterboxd prepends
    if review.lower().startswith(title.lower()):
        review = review[len(title):].lstrip(" ,.\n")
    review = review[:500].strip()  # cap at 500 chars

    return {
        "title": str(title),
        "year": str(year),
        "rating_numeric": float(rating_raw) if rating_raw else None,
        "rating_stars": numeric_to_stars(rating_raw),
        "watched_date": str(watched_date),
        "rewatch": rewatch,
        "poster": poster,
        "review": review,
        "link": link,
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(os.path.join(root, "_config.yml"))

    lb_cfg = config.get("letterboxd", {})
    username = lb_cfg.get("username", "")
    if not username:
        raise ValueError("letterboxd.username not set in _config.yml")

    rss_url = f"https://letterboxd.com/{username}/rss/"
    print(f"Fetching {rss_url}")
    feed = feedparser.parse(rss_url)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse RSS: {feed.bozo_exception}")

    existing = load_existing(os.path.join(root, "_data", "letterboxd.yml"))
    existing_links = {f["link"] for f in existing}

    new_films = []
    for entry in feed.entries:
        film = parse_entry(entry)
        # Skip Letterboxd lists and diary entries without a watch date
        if not film["watched_date"] or "/list/" in film["link"]:
            continue
        if film["link"] and film["link"] not in existing_links:
            new_films.append(film)
            existing_links.add(film["link"])

    all_films = new_films + existing
    all_films.sort(key=lambda f: f.get("watched_date", "") or "", reverse=True)

    output = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "films": all_films,
    }

    data_path = os.path.join(root, "_data", "letterboxd.yml")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w") as f:
        yaml.dump(output, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Done. {len(new_films)} new, {len(existing)} existing → {len(all_films)} total films.")


if __name__ == "__main__":
    main()
