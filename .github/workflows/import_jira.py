import os
import requests

JIRA_URL = os.environ["JIRA_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

print("=== DIAGNOSTIC TEST ===\n")

# 1. Check ML
print("1. Searching for 'project = ML'...")
r1 = requests.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    json={"jql": "project = ML", "maxResults": 1, "fields": ["summary"]}
)
if r1.status_code == 200:
    data1 = r1.json()
    print(f"   Total issues in ML: {data1.get('total', 'ERROR')}")
else:
    print(f"   Error: {r1.text[:300]}")

print("\n2. Searching for ANY issues in your Jira...")
r2 = requests.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    json={"jql": "order by created DESC", "maxResults": 10, "fields": ["summary"]}
)
if r2.status_code == 200:
    data2 = r2.json()
    print(f"   Total issues found: {data2.get('total', 'ERROR')}")
    for issue in data2.get("issues", []):
        print(f"   -> {issue['key']}: {issue['fields']['summary'][:50]}")
    if not data2.get("issues"):
        print("   -> NO ISSUES FOUND AT ALL.")
else:
    print(f"   Error: {r2.text[:300]}")

print("\n=== DONE ===")
