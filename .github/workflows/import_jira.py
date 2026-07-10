import os
import requests

JIRA_URL = os.environ["JIRA_URL"].strip().rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"].strip()
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"].strip()

print(f"URL: {JIRA_URL}")
print(f"Email: {JIRA_EMAIL}")
print(f"Token starts with: {JIRA_API_TOKEN[:5]}")

r = requests.get(
    f"{JIRA_URL}/rest/api/3/myself",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json"}
)

print(f"\nStatus: {r.status_code}")
if r.status_code == 200:
    print(f"✅ SUCCESS! Logged in as: {r.json().get('displayName')}")
else:
    print(f"❌ FAILED: {r.text}")
