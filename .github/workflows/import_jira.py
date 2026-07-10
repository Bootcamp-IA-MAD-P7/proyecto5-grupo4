import os
import requests
import time

JIRA_URL = os.environ["JIRA_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

OWNER = "Bootcamp-IA-MAD-P7"
REPO = "proyecto5-grupo4"

github_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

print(f"Calling: {JIRA_URL}/rest/api/3/search")
print("=== Fetching from Jira ===")

# POST request (GET is deprecated in Jira Cloud)
response = requests.post(
    f"{JIRA_URL}/rest/api/3/search",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
    json={
        "jql": "project = ML",
        "maxResults": 100,
        "fields": ["summary"],
    },
)

print(f"Status: {response.status_code}")

if response.status_code != 200:
    print(f"Error: {response.text[:500]}")
    exit(1)

data = response.json()
issues = data.get("issues", [])
print(f"Found {len(issues)} issues")

created = 0
for issue in issues:
    key = issue["key"]
    title = issue["fields"]["summary"]
    
    r = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/issues",
        headers=github_headers,
        json={"title": f"[{key}] {title}"},
    )
    
    if r.status_code == 201:
        print(f"✓ {key} - {title}")
        created += 1
    else:
        print(f"✗ {key} - {r.status_code}: {r.text[:100]}")
    
    time.sleep(1)

print(f"\n=== Done: {created} created ===")
