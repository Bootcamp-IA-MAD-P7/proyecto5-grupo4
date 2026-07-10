import os
import requests

JIRA_URL = os.environ["JIRA_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

print("=== DIAGNOSTIC TEST ===\n")

# Test 1: How many issues does Jira think are in project ML?
print("1. Searching for 'project = ML'...")
r1 = requests.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    json={"jql": "project = ML", "maxResults": 0, "fields": []}
)
if r1.status_code == 200:
    data1 = r1.json()
    print(f"   Total issues in ML: {data1.get('total', 'ERROR')}")
    print(f"   Warning message: {data1.get('warningMessages', 'None')}")
else:
    print(f"   Error: {r1.text[:200]}")

print("\n2. Searching for ANY issues you have access to...")
# Test 2: Just find ANY issue in the whole system for this user
r2 = requests.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    json={"jql": "assignee = currentUser() ORDER BY created DESC", "maxResults": 5, "fields": ["summary"]}
)
if r2.status_code == 200:
    data2 = r2.json()
    print(f"   Total issues assigned to you: {data2.get('total', 'ERROR')}")
    for issue in data2.get("issues", []):
        print(f"   -> {issue['key']}: {issue['fields']['summary'][:50]}")
else:
    print(f"   Error: {r2.text[:200]}")

print("\n=== DONE ===")
