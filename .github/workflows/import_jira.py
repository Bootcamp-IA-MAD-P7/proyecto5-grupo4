import os
import requests

JIRA_URL = os.environ["JIRA_URL"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

jira = requests.Session()
jira.auth = (JIRA_EMAIL, JIRA_API_TOKEN)
jira.headers.update({"Accept": "application/json"})

print(f"JIRA_URL: {JIRA_URL}")
print()

# 1. Check if we can connect at all
print("=== Test 1: Basic connectivity ===")
r = jira.get(f"{JIRA_URL}/rest/api/3/myself")
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Logged in as: {r.json().get('displayName', 'unknown')}")
else:
    print(r.text)
print()

# 2. List projects
print("=== Test 2: List projects ===")
r = jira.get(f"{JIRA_URL}/rest/api/3/project")
print(f"Status: {r.status_code}")
if r.status_code == 200:
    for p in r.json():
        print(f"  {p['key']} - {p['name']}")
else:
    print(r.text)
print()

# 3. Try different JQL formats
print("=== Test 3: JQL queries ===")

queries = [
    "project = ML",
    "project = ML ORDER BY created ASC",
    "project=ML",
]

for q in queries:
    print(f"\nQuery: {q}")
    r = jira.post(
        f"{JIRA_URL}/rest/api/3/search/jql",
        json={"jql": q, "maxResults": 5, "fields": ["summary"]},
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Total: {data.get('total', 0)}")
        for issue in data.get("issues", [])[:3]:
            print(f"    - {issue['key']}: {issue['fields']['summary'][:50]}")
    else:
        print(f"  Error: {r.text[:200]}")
