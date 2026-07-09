import os
import requests

JIRA_URL = os.environ["JIRA_URL"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT_KEY = os.environ["JIRA_PROJECT_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

OWNER = "Bootcamp-IA-MAD-P7"
REPO = "proyecto5-grupo4"

jira = requests.Session()
jira.auth = (JIRA_EMAIL, JIRA_API_TOKEN)
jira.headers.update({"Accept": "application/json"})

github_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

start_at = 0
max_results = 100

while True:
    response = jira.get(
        f"{JIRA_URL}/rest/api/3/jql",
        params={
            "jql": f"project={JIRA_PROJECT_KEY} ORDER BY created ASC",
            "startAt": start_at,
            "maxResults": max_results,
            "fields": "summary",
        },
    )

    response.raise_for_status()

    data = response.json()
    issues = data.get("issues", [])

    if not issues:
        break

    for issue in issues:
        key = issue["key"]
        title = issue["fields"]["summary"]

        r = requests.post(
            f"https://api.github.com/repos/{OWNER}/{REPO}/issues",
            headers=github_headers,
            json={
                "title": title,
            },
        )

        if r.status_code == 201:
            print(f"✓ {key} → {title}")
        else:
            print(f"✗ {key}")
            print(r.text)

    start_at += max_results
