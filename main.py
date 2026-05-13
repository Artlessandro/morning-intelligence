"""Morning Intelligence — daily personalised brief generator.

Fetches the freshest tech / startup / AI stories from Hacker News and NewsAPI,
asks Claude Haiku to distil them into a five-section brief, and emails it via Gmail.
"""
from __future__ import annotations

import os
import smtplib
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

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
TARGET_STORIES = 18
WIB = timezone(timedelta(hours=7))


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


def fetch_newsapi_stories() -> list[dict]:
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {
        "q": "AI OR startup OR \"venture capital\" OR \"product launch\" OR \"design tool\" OR SaaS",
        "language": "en",
        "sortBy": "publishedAt",
        "from": today_utc,
        "pageSize": 40,
        "apiKey": NEWSAPI_KEY,
    }
    r = requests.get(NEWSAPI_URL, params=params, timeout=15)
    r.raise_for_status()
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
            "published_at": a.get("publishedAt", ""),
        })
    return stories


def select_stories(hn: list[dict], na: list[dict], target: int = TARGET_STORIES) -> list[dict]:
    half = target // 2
    return (hn[:half] + na[: target - half])[:target]


def build_prompt(stories: list[dict]) -> str:
    lines = []
    for i, s in enumerate(stories, 1):
        line = f"{i}. [{s['source']}] {s['title']}"
        desc = s.get("description")
        if desc:
            line += f" — {desc}"
        if s.get("url"):
            line += f" ({s['url']})"
        lines.append(line)
    story_block = "\n".join(lines)
    return (
        "You are writing a morning intelligence brief for a product builder and "
        "design studio founder. The reader cares about: AI, startups, design tooling, "
        "product launches, funding flows, and emerging consumer / enterprise patterns.\n\n"
        f"Below are {len(stories)} stories from the last 24 hours, drawn from Hacker News and NewsAPI.\n\n"
        f"STORIES:\n{story_block}\n\n"
        "Write the brief as clean HTML using these EXACT five section headers (<h2>):\n\n"
        "1. <h2>Signal of the day</h2> — one paragraph on the single biggest story. "
        "Explain what happened and why it specifically matters to product builders and design studio founders.\n\n"
        "2. <h2>3 problems worth solving</h2> — a <ul> with three <li> items. "
        "Each is real friction or a complaint surfaced in today's stories, framed as an opportunity (not a summary).\n\n"
        "3. <h2>Where money is moving</h2> — a <ul> with 2–3 tight bullets on funding rounds, tool launches, or categories gaining traction.\n\n"
        "4. <h2>One build idea</h2> — a single concrete product or service inspired by today's signals. "
        "Include: what it is, who would pay for it, and roughly what they'd pay.\n\n"
        "5. <h2>One sentence to carry</h2> — a sharp, thought-provoking single sentence or question for the day. "
        "Wrap it in a <blockquote>.\n\n"
        "Rules:\n"
        "- Output ONLY the HTML body content. No <html>, <head>, <body>, <style> tags. No markdown code fences.\n"
        "- Write in confident, plain English. Avoid hype words like 'revolutionary' or 'game-changer'.\n"
        "- Keep it tight — readable in under 90 seconds.\n"
        "- Only use stories from the list above. Do not invent facts or sources.\n"
    )


def generate_brief(stories: list[dict]) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": build_prompt(stories)}],
    )
    return msg.content[0].text.strip()


def wrap_html(body_html: str) -> str:
    today = datetime.now(timezone.utc).astimezone(WIB).strftime("%A, %d %B %Y")
    return (
        "<!doctype html>\n"
        "<html><body style=\"font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
        "max-width:640px;margin:0 auto;padding:24px;color:#111;line-height:1.55;\">"
        f"<p style=\"color:#666;margin:0 0 4px;font-size:13px;\">Morning Intelligence · {today}</p>"
        "<h1 style=\"margin:0 0 24px;font-size:22px;\">Today's brief</h1>"
        f"{body_html}"
        "<hr style=\"border:none;border-top:1px solid #eee;margin:32px 0 12px;\">"
        "<p style=\"color:#999;font-size:12px;\">Generated daily at 9am WIB by Morning Intelligence.</p>"
        "</body></html>"
    )


def send_email(html: str) -> None:
    today_wib = datetime.now(timezone.utc).astimezone(WIB).strftime("%a %d %b")
    msg = EmailMessage()
    msg["Subject"] = f"Morning Intelligence — {today_wib}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg.set_content("Your client cannot display HTML. Open in a modern email client to read today's brief.")
    msg.add_alternative(html, subtype="html")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD.replace(" ", ""))
        smtp.send_message(msg)


def main() -> int:
    print("Fetching Hacker News…")
    hn = fetch_hn_stories()
    print(f"  {len(hn)} HN stories in the last 24h")

    print("Fetching NewsAPI…")
    na = fetch_newsapi_stories()
    print(f"  {len(na)} NewsAPI stories")

    stories = select_stories(hn, na)
    if not stories:
        print("No fresh stories found — aborting.", file=sys.stderr)
        return 1
    print(f"Sending {len(stories)} stories to Claude ({CLAUDE_MODEL})…")

    body = generate_brief(stories)
    html = wrap_html(body)

    print("Sending email…")
    send_email(html)
    print(f"Sent to {RECIPIENT_EMAIL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
