import os
import requests
import time

JIRA_URL = os.environ["JIRA_URL"].strip().rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"].strip()
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"].strip()
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

OWNER = "Bootcamp-IA-MAD-P7"
REPO = "proyecto5-grupo4"

jira_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

github_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

print("=== Importing Jira Issues to GitHub ===")

# 1. Fetch from Jira
print("Fetching from Jira...")
response = requests.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers=jira_headers,
    json={
        "jql": "project = ML",
        "maxResults": 100,
        "fields": ["summary"],
    },
)

if response.status_code != 200:
    print(f"Jira Error: {response.text}")
    exit(1)

data = response.json()
issues = data.get("issues", [])
print(f"Found {len(issues)} issues in Jira.\n")

# 2. Create in GitHub
created = 0
for issue in issues:
    key = issue["key"]
    title = issue["fields"]["summary"]
    
    r = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/issues",
        headers=github_headers,
        json={"title": f"[{key}] {title}"}
    )
    
    if r.status_code == 201:
        print(f"✓ Created: {key} - {title}")
        created += 1
    else:
        print(f"✗ Failed: {key} - Error {r.status_code}: {r.text[:100]}")
        
    time.sleep(1) # Avoid GitHub rate limits

print(f"\n=== DONE: {created} issues created in GitHub ===")
