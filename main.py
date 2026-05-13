"""Morning Intelligence — daily personalised brief generator.

Fetches the freshest tech / startup / AI / funding / design / macro stories from
Hacker News and NewsAPI, asks Claude Haiku to distil them into a structured brief
with longitudinal context (yesterday's top stories), and emails it via Gmail.
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import quote

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NEWSAPI_KEY = os.environ["NEWSAPI_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ["RECIPIENT_EMAIL"]

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
HN_NEWSTORIES_URL = "https://hacker-news.firebaseio.com/v0/newstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

HOURS_WINDOW = 24
HN_FETCH_LIMIT = 80
TARGET_GENERAL = 14
TARGET_FUNDING = 8
TARGET_DESIGN = 6
TARGET_MACRO = 5
WIB = timezone(timedelta(hours=7))

STATE_PATH = Path("state/last_brief.json")

GENERAL_QUERY = (
    'AI OR startup OR "venture capital" OR "product launch" OR "design tool" OR SaaS'
)
FUNDING_QUERY = (
    '"Series A" OR "Series B" OR "Series C" OR "Series D" OR "seed round" '
    'OR "raises" OR "funding round" OR "valuation" OR "acquisition"'
)
DESIGN_QUERY = (
    '"design tool" OR "design system" OR Figma OR Framer OR Webflow '
    'OR typography OR "product design" OR "UX research" OR "design agency"'
)
MACRO_QUERY = (
    '"interest rates" OR "Federal Reserve" OR "stock market" OR IPO '
    'OR earnings OR inflation OR recession OR "tech layoffs"'
)


def fetch_hn_stories() -> list[dict]:
    cutoff = time.time() - HOURS_WINDOW * 3600
    ids = requests.get(HN_NEWSTORIES_URL, timeout=15).json()[:HN_FETCH_LIMIT]
    stories: list[dict] = []
    for sid in ids:
        try:
            item = requests.get(HN_ITEM_URL.format(id=sid), timeout=10).json()
        except requests.RequestException:
            continue
        if not item or item.get("type") != "story":
            continue
        if item.get("time", 0) < cutoff:
            continue
        title = item.get("title")
        if not title:
            continue
        stories.append({
            "source": "Hacker News",
            "title": title,
            "url": item.get("url") or f"https://news.ycombinator.com/item?id={sid}",
            "score": item.get("score", 0),
            "published_at": item.get("time", 0),
        })
    stories.sort(key=lambda s: (s["score"], s["published_at"]), reverse=True)
    return stories


def fetch_newsapi(query: str, page_size: int) -> list[dict]:
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "from": today_utc,
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        r = requests.get(NEWSAPI_URL, params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"  NewsAPI fetch failed ({query[:40]}...): {exc}", file=sys.stderr)
        return []
    data = r.json()
    articles = data.get("articles", [])
    stories: list[dict] = []
    for a in articles:
        title = a.get("title")
        if not title or title == "[Removed]":
            continue
        stories.append({
            "source": (a.get("source") or {}).get("name") or "NewsAPI",
            "title": title,
            "url": a.get("url", ""),
            "description": a.get("description") or "",
            "image": a.get("urlToImage") or "",
            "published_at": a.get("publishedAt", ""),
        })
    return stories


def dedupe_by_url(buckets: list[list[dict]]) -> None:
    """Remove URLs that appear in earlier buckets from later ones, in place."""
    seen: set[str] = set()
    for bucket in buckets:
        kept = []
        for s in bucket:
            url = s.get("url", "")
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            kept.append(s)
        bucket[:] = kept


def format_story_block(stories: list[dict], start_index: int = 1) -> str:
    lines = []
    for i, s in enumerate(stories, start_index):
        line = f"{i}. [{s['source']}] {s['title']}"
        desc = s.get("description")
        if desc:
            line += f" -- {desc}"
        if s.get("url"):
            line += f" ({s['url']})"
        lines.append(line)
    return "\n".join(lines)


def format_yesterday_block(state: dict | None) -> str:
    if not state or not state.get("top_stories"):
        return ""
    lines = []
    for i, s in enumerate(state["top_stories"][:6], 1):
        lines.append(f"{i}. {s.get('title','')}")
    return "\n".join(lines)


def build_prompt(
    general: list[dict],
    funding: list[dict],
    design: list[dict],
    macro: list[dict],
    yesterday: dict | None,
) -> str:
    general_block = format_story_block(general, 1)
    funding_block = format_story_block(funding, len(general) + 1)
    design_block = format_story_block(design, len(general) + len(funding) + 1)
    macro_block = format_story_block(
        macro, len(general) + len(funding) + len(design) + 1
    )

    yesterday_section = ""
    yesterday_spec = ""
    yb = format_yesterday_block(yesterday)
    if yb:
        yesterday_section = (
            "\n\nYESTERDAY'S TOP STORIES (for 'What changed since yesterday'):\n"
            f"{yb}\n"
        )
        yesterday_spec = (
            "<h2>What changed since yesterday</h2>: a <ul> with 2 short bullets. "
            "Each bullet identifies a story from yesterday that has evolved, been "
            "contradicted, or gained a new development today. Hyperlink the relevant "
            "noun phrase to today's source URL. If nothing connects, write one bullet "
            "noting a fresh thread that did not exist yesterday."
        )

    section_specs = [
        "<h2>Signal of the day</h2>: one paragraph on the single biggest story from "
        "the GENERAL stories below. Explain what happened and why it specifically "
        "matters to product builders and design studio founders. Hyperlink the "
        "company or product name to its source URL.",
        "<h2>Pattern emerging</h2>: one paragraph naming a recurring theme across "
        "3 or more stories today, from any bucket. Make it a take, not a summary. "
        "Hyperlink at least two supporting stories inline.",
        yesterday_spec,
        "<h2>Three problems worth solving</h2>: a <ul> with three <li> items. Each "
        "is real friction or a complaint surfaced in today's stories, framed as an "
        "opportunity. Each bullet must hyperlink the originating story.",
        "<h2>Where money is moving</h2>: a <ul> with 3 to 4 tight bullets on funding "
        "rounds, valuations, acquisitions, or categories gaining capital. Prefer "
        "items from the FUNDING bucket. Every bullet hyperlinks the source.",
        "<h2>Market context</h2>: a <ul> with 2 short bullets framing today against "
        "the macro lens, using the MACRO bucket: rates, IPOs, big-cap earnings, "
        "layoffs, or sector rotation that affects builders. Every bullet hyperlinks "
        "the source.",
        "<h2>Design pulse</h2>: a <ul> with 2 bullets on the design industry: tools, "
        "agencies, typography, design system shifts, or notable launches. Prefer "
        "items from the DESIGN bucket. Every bullet hyperlinks the source.",
        "<h2>One build idea</h2>: a single concrete product or service inspired by "
        "today's signals. Include what it is, who would pay for it, and roughly what "
        "they would pay. Hyperlink the story or stories that inspired it.",
        "<h2>People worth reaching this week</h2>: a <ul> with 3 to 6 <li> items, "
        "each a real named person mentioned in today's stories: a founder who raised, "
        "an exec who moved roles, a builder who shipped, an investor leading a round. "
        "For each item, use EXACTLY this format:\n"
        "<li><strong>Full Name</strong>, role at Company. One short sentence on why "
        "they are worth reaching, tied to the reader's design studio and product "
        "work. <a href=\"https://www.linkedin.com/search/results/people/?keywords=NAME%20COMPANY\">Find on LinkedIn</a> "
        "&middot; <a href=\"SOURCE_URL\">Source</a></li>\n"
        "Build the LinkedIn URL by URL-encoding the person's full name plus their "
        "company, joining with %20 (e.g. 'Jane%20Doe%20Acme%20AI'). Only include "
        "people who are actually named in the story title or description. Never "
        "invent a name. If fewer than 3 people are named today, output only as many "
        "as exist, with one final <li> reading 'Light day for named individuals.'",
        "<h2>One sentence to carry</h2>: a sharp, thought-provoking single sentence "
        "or question for the day. Wrap it in a <blockquote>.",
    ]
    section_specs = [s for s in section_specs if s]
    numbered = [f"{i + 1}. {s}" for i, s in enumerate(section_specs)]

    return (
        "You are writing a morning intelligence brief for a product builder and "
        "design studio founder. The reader cares about: AI, startups, design tooling, "
        "product launches, funding flows, macro signals, and emerging consumer / "
        "enterprise patterns.\n\n"
        "Below are story buckets from the last 24 hours. Each story is numbered "
        "globally across buckets so you can reference one by number.\n\n"
        f"GENERAL STORIES:\n{general_block}\n\n"
        f"FUNDING STORIES:\n{funding_block}\n\n"
        f"DESIGN STORIES:\n{design_block}\n\n"
        f"MACRO STORIES:\n{macro_block}"
        f"{yesterday_section}\n\n"
        "FIRST, on the very first two lines of your output, write two HTML comments. "
        "Line 1 is the global story number you picked for 'Signal of the day'. "
        "Line 2 is two concrete visual nouns for a header image fallback. Format:\n"
        "<!-- SIGNAL: 5 -->\n"
        "<!-- HERO: keyword1,keyword2 -->\n"
        "Use concrete visual nouns (e.g. 'chip,circuit', 'skyscraper,city', "
        "'sketchbook,desk'), not abstract concepts.\n\n"
        "THEN write the brief as clean HTML using these EXACT section headers "
        "(<h2>) in order:\n\n"
        + "\n\n".join(numbered)
        + "\n\n"
        "Rules:\n"
        "- Output ONLY the HTML body content. No <html>, <head>, <body>, <style> "
        "tags. No markdown code fences.\n"
        "- HYPERLINK every story reference. When you mention a company, product, "
        "founder, or fact from a story, wrap the most relevant noun phrase in "
        "<a href=\"STORY_URL\">...</a> using the URL provided in parentheses after "
        "that story in the buckets above. Every bullet and paragraph that draws "
        "from a story must contain at least one such link.\n"
        "- DO NOT use em-dashes (the long dash) anywhere. Use commas, full stops, "
        "or parentheses for the same effect. This is strict.\n"
        "- DO NOT use en-dashes between words for the same reason. Numeric ranges "
        "like 2024-2025 are fine.\n"
        "- Separate paragraphs and ideas with their own <p> tags. Do not pack "
        "multiple distinct points into one long <p>. Short paragraphs read better.\n"
        "- Write in confident, plain English. Avoid hype words like 'revolutionary' "
        "or 'game-changer'.\n"
        "- Keep each section tight. The whole brief should be readable in under "
        "2 minutes.\n"
        "- Only use stories from the buckets above. Do not invent facts, people, "
        "or sources.\n"
    )


def generate_brief(prompt: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


HERO_RE = re.compile(r"<!--\s*HERO:\s*([^>]+?)\s*-->", re.IGNORECASE)
SIGNAL_RE = re.compile(r"<!--\s*SIGNAL:\s*#?(\d+)\s*-->", re.IGNORECASE)


def extract_markers(body: str) -> tuple[str, list[str], int | None]:
    """Strip leading HERO/SIGNAL comments. Return (cleaned_body, keywords, signal_index)."""
    keywords: list[str] = ["technology", "city"]
    signal: int | None = None

    m = HERO_RE.search(body)
    if m:
        parsed = [k.strip().lower() for k in m.group(1).split(",") if k.strip()]
        if parsed:
            keywords = parsed[:3]
        body = HERO_RE.sub("", body, count=1)

    m = SIGNAL_RE.search(body)
    if m:
        try:
            signal = int(m.group(1))
        except ValueError:
            signal = None
        body = SIGNAL_RE.sub("", body, count=1)

    return body.lstrip(), keywords, signal


def pick_hero(
    keywords: list[str],
    signal_index: int | None,
    story_index: dict[int, dict],
) -> tuple[str, str]:
    """Return (image_url, alt_text). Prefer the signal story's own image; fall back to Loremflickr."""
    if signal_index and signal_index in story_index:
        story = story_index[signal_index]
        img = story.get("image", "")
        if img and img.startswith(("http://", "https://")):
            alt = (story.get("title") or ", ".join(keywords))[:140]
            return img, alt

    encoded = ",".join(quote(k, safe="") for k in keywords)
    return f"https://loremflickr.com/1200/500/{encoded}", ", ".join(keywords)


def build_story_index(*buckets: list[dict]) -> dict[int, dict]:
    flat: dict[int, dict] = {}
    i = 1
    for bucket in buckets:
        for s in bucket:
            flat[i] = s
            i += 1
    return flat


def clean_dashes(html: str) -> str:
    """Replace em-dashes and prose en-dashes with comma+space."""
    html = re.sub(r"\s*—\s*", ", ", html)
    html = re.sub(r"(?<=[A-Za-z\)])\s+–\s+(?=[A-Za-z\(])", ", ", html)
    return html


def wrap_html(body_html: str, hero_url: str, hero_alt: str) -> str:
    today = datetime.now(timezone.utc).astimezone(WIB).strftime("%A, %d %B %Y")
    return (
        "<!doctype html>\n"
        "<html><body style=\"font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
        "max-width:680px;margin:0 auto;padding:32px 24px;color:#111;line-height:1.7;"
        "font-size:16px;background:#fff;\">"
        f"<p style=\"color:#777;margin:0 0 6px;font-size:13px;letter-spacing:0.02em;\">"
        f"Morning Intelligence &middot; {today}</p>"
        "<h1 style=\"margin:0 0 24px;font-size:24px;font-weight:600;\">Today's brief</h1>"
        f"<img src=\"{hero_url}\" alt=\"{hero_alt}\" "
        "style=\"width:100%;height:280px;object-fit:cover;object-position:center;"
        "display:block;border-radius:8px;margin:0 0 36px;background:#f4f4f4;\">"
        "<style>"
        "h2{font-size:18px;font-weight:600;margin:40px 0 14px;color:#111;}"
        "p{margin:0 0 18px;}"
        "ul{margin:0 0 24px;padding-left:22px;}"
        "li{margin:0 0 12px;}"
        "blockquote{margin:24px 0;padding:16px 20px;border-left:3px solid #111;"
        "background:#fafafa;font-style:italic;color:#222;}"
        "a{color:#0a66c2;text-decoration:none;}"
        "</style>"
        "<div class=\"brief\">"
        f"{body_html}"
        "</div>"
        "<hr style=\"border:none;border-top:1px solid #eee;margin:40px 0 12px;\">"
        "<p style=\"color:#999;font-size:12px;\">Generated daily at 9am WIB by Morning Intelligence.</p>"
        "</body></html>"
    )


def send_email(html: str) -> None:
    today_wib = datetime.now(timezone.utc).astimezone(WIB).strftime("%a %d %b")
    msg = EmailMessage()
    msg["Subject"] = f"Morning Intelligence, {today_wib}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg.set_content("Your client cannot display HTML. Open in a modern email client to read today's brief.")
    msg.add_alternative(html, subtype="html")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD.replace(" ", ""))
        smtp.send_message(msg)


def load_yesterday() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  Could not read yesterday state: {exc}", file=sys.stderr)
        return None


def save_today(general: list[dict], funding: list[dict]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d"),
        "top_stories": [
            {"title": s["title"], "url": s.get("url", ""), "source": s.get("source", "")}
            for s in (general[:5] + funding[:3])
        ],
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    print("Fetching Hacker News...")
    hn = fetch_hn_stories()
    print(f"  {len(hn)} HN stories in the last 24h")

    print("Fetching NewsAPI (general)...")
    na_general = fetch_newsapi(GENERAL_QUERY, 40)
    print(f"  {len(na_general)} general stories")

    print("Fetching NewsAPI (funding)...")
    funding = fetch_newsapi(FUNDING_QUERY, 25)
    print(f"  {len(funding)} funding stories")

    print("Fetching NewsAPI (design)...")
    design = fetch_newsapi(DESIGN_QUERY, 20)
    print(f"  {len(design)} design stories")

    print("Fetching NewsAPI (macro)...")
    macro = fetch_newsapi(MACRO_QUERY, 15)
    print(f"  {len(macro)} macro stories")

    half = TARGET_GENERAL // 2
    general = hn[:half] + na_general[: TARGET_GENERAL - half]
    funding = funding[:TARGET_FUNDING]
    design = design[:TARGET_DESIGN]
    macro = macro[:TARGET_MACRO]

    dedupe_by_url([general, funding, design, macro])

    if not general:
        print("No fresh general stories found, aborting.", file=sys.stderr)
        return 1

    yesterday = load_yesterday()
    if yesterday:
        print(f"  loaded yesterday state ({yesterday.get('date','?')})")
    else:
        print("  no yesterday state, skipping continuity section")

    total = len(general) + len(funding) + len(design) + len(macro)
    print(f"Sending {total} stories to Claude ({CLAUDE_MODEL})...")
    prompt = build_prompt(general, funding, design, macro, yesterday)
    body = generate_brief(prompt)

    body, keywords, signal_index = extract_markers(body)
    body = clean_dashes(body)

    story_index = build_story_index(general, funding, design, macro)
    hero_url, hero_alt = pick_hero(keywords, signal_index, story_index)
    print(f"  signal story #{signal_index}, hero keywords {keywords}")
    print(f"  hero image: {hero_url[:80]}{'...' if len(hero_url) > 80 else ''}")

    html = wrap_html(body, hero_url, hero_alt)

    print("Sending email...")
    send_email(html)
    print(f"Sent to {RECIPIENT_EMAIL}")

    save_today(general, funding)
    print(f"Saved state to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
