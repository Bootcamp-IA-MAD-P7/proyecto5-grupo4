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

print(f"=== Importing from Jira project: {JIRA_PROJECT_KEY} ===")

response = jira.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    json={
        "jql": f"project = {JIRA_PROJECT_KEY} ORDER BY cf[10019] ASC",
        "maxResults": 100,
        "fields": ["summary"],
    },
)

if response.status_code != 200:
    print(f"Jira error {response.status_code}: {response.text}")
    exit(1)

issues = response.json().get("issues", [])
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
        print(f"✗ {key} - Error {r.status_code}: {r.text[:200]}")
    
    time.sleep(1)

print(f"\n=== Done: {created} issues created ===")
