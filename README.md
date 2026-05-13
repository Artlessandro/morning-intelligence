# Morning Intelligence

A daily, personalised morning brief for product builders and design studio founders.

Every day at **9am WIB (02:00 UTC)**, a GitHub Actions workflow runs `main.py`, which:

1. Pulls the freshest tech / startup / AI stories from the last 24 hours from **Hacker News** and **NewsAPI**.
2. Sends 15–20 of them to **Claude Haiku** with a tight prompt.
3. Receives back a five-section brief and emails it via **Gmail SMTP**.

The brief has the same shape every day:

- **Signal of the day** — the single biggest story, and why it matters to product builders.
- **3 problems worth solving** — frictions surfaced in today's news, framed as opportunities.
- **Where money is moving** — funding, launches, and categories gaining traction.
- **One build idea** — a concrete product/service idea with target buyer and rough price.
- **One sentence to carry** — a thought to sit with for the day.

## Folder structure

```
.
├── main.py                    # The brief generator
├── requirements.txt           # Python dependencies
├── .env.example               # Template for local secrets
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── run.yml            # Daily cron job
```

## Setting it up from scratch

### 1. Install prerequisites

- **Python 3.12+** — https://www.python.org/downloads/ (tick "Add to PATH")
- **Git** — https://git-scm.com/download/win
- **VS Code** (optional but recommended) with the Python extension

### 2. Get API credentials

You will need five secrets:

| Secret | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |
| `NEWSAPI_KEY` | https://newsapi.org/register (free tier) |
| `GMAIL_ADDRESS` | Your Gmail address |
| `GMAIL_APP_PASSWORD` | https://myaccount.google.com/apppasswords (requires 2FA) |
| `RECIPIENT_EMAIL` | Where to send the brief |

### 3. Local testing

```powershell
git clone https://github.com/<your-username>/morning-intelligence.git
cd morning-intelligence
pip install -r requirements.txt
copy .env.example .env   # then fill in real values
python main.py
```

If everything is wired correctly, you'll receive an email within ~20 seconds.

### 4. Schedule it on GitHub Actions

1. Push this repo to GitHub (private recommended).
2. Go to **Settings → Secrets and variables → Actions** and add all five secrets above with the exact names shown.
3. The workflow in `.github/workflows/run.yml` runs automatically at **02:00 UTC** every day.
4. You can also trigger it manually: **Actions** tab → "Morning Intelligence" → **Run workflow**.

## Cost

- **Anthropic** — Claude Haiku 4.5 at this prompt size is roughly a fraction of a cent per run, well under $1/month.
- **NewsAPI** — free tier (100 requests/day, more than enough).
- **GitHub Actions** — free tier covers daily 1-minute jobs on public *and* private repos for personal accounts.
- **Gmail** — free.

## Notes

- Secrets are read from environment variables only. Nothing is hardcoded.
- For local runs, `python-dotenv` loads them from `.env`. On Actions, they come from GitHub Secrets.
- The Gmail App Password may be displayed with spaces — those are stripped at login time.
