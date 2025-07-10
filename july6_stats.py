import os
from datetime import datetime, timedelta
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Set GITHUB_TOKEN for API access")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

USERNAME = "mhcoen"
TARGET_DATE = "2025-07-06"
NEXT_DATE = "2025-07-07"

def fetch_events(page=1):
    url = f"https://api.github.com/users/{USERNAME}/events"
    resp = requests.get(url, headers=HEADERS, params={"per_page": 100, "page": page})
    resp.raise_for_status()
    return resp.json()

def parse_events(events):
    stats = {"Commits": 0, "PR Opened": 0, "PR Merged": 0, "PR Reviewed": 0, "Issue Opened": 0, "Issue Comment": 0}
    for ev in events:
        ts = ev["created_at"][:10]
        if ts != TARGET_DATE:
            continue
        t = ev["type"]
        if t == "PushEvent":
            stats["Commits"] += len(ev["payload"].get("commits", []))
        elif t == "PullRequestEvent":
            action = ev["payload"]["action"]
            if action == "opened":
                stats["PR Opened"] +=1
            elif action == "closed" and ev["payload"]["pull_request"]["merged"]:
                stats["PR Merged"] +=1
        elif t == "PullRequestReviewEvent":
            stats["PR Reviewed"] +=1
        elif t == "IssuesEvent":
            if ev["payload"]["action"] == "opened":
                stats["Issue Opened"] +=1
        elif t == "IssueCommentEvent":
            stats["Issue Comment"] +=1
    return stats

# iterate through pages
page = 1
aggregate = {"Commits":0,"PR Opened":0,"PR Merged":0,"PR Reviewed":0,"Issue Opened":0,"Issue Comment":0}
while True:
    events = fetch_events(page)
    if not events:
        break
    stats = parse_events(events)
    for k in aggregate:
        aggregate[k] += stats[k]
    page += 1

print("Your July 6, 2025 contribution breakdown:")
for k, v in aggregate.items():
    print(f"{k}: {v}")
    
