import os
import requests
import time

JIRA_URL = os.environ["JIRA_URL"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT_KEY = os.environ["JIRA_PROJECT_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

OWNER = "Bootcamp-IA-MAD-P7"
REPO = "proyecto5-grupo4"

jira = requests.Session()
jira.auth = (JIRA_EMAIL, JIRA_API_TOKEN)
jira.headers.update({
    "Accept": "application/json",
    "Content-Type": "application/json",
})

github_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

start_at = 0
max_results = 50
total_created = 0

print(f"=== Starting import to {OWNER}/{REPO} ===")

while True:
    print(f"\n[Page] startAt={start_at}")
    
    response = jira.post(
        f"{JIRA_URL}/rest/api/3/search/jql",  # Your working endpoint
        json={
            "jql": f"project = {JIRA_PROJECT_KEY} ORDER BY cf[10019] ASC",
            "startAt": start_at,  # FIX: Add pagination
            "maxResults": max_results,
            "fields": ["summary"],
        },
    )

    if response.status_code != 200:
        print(f"JIRA ERROR: {response.status_code} - {response.text}")
        break

    data = response.json()
    issues = data.get("issues", [])
    total = data.get("total", 0)
    
    print(f"Jira returned {len(issues)} issues (total: {total})")

    if not issues:
        print("No more issues")
        break

    for issue in issues:
        key = issue["key"]
        title = issue["fields"]["summary"]

        print(f"Creating: {key} - {title[:50]}...", end=" ")
        
        r = requests.post(
            f"https://api.github.com/repos/{OWNER}/{REPO}/issues",
            headers=github_headers,
            json={
                "title": f"[{key}] {title}",
            },
        )

        if r.status_code == 201:
            print("✓")
            total_created += 1
        else:
            print(f"✗ ({r.status_code}: {r.text[:100]})")
        
        time.sleep(1)  # Avoid rate limit

    start_at += max_results  # FIX: Move to next page
    
    if start_at >= total:  # FIX: Stop when done
        break

print(f"\n=== FINISHED: {total_created} issues created ===")
