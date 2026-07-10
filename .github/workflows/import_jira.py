import os
import requests

JIRA_URL = os.environ["JIRA_URL"].strip().rstrip("/")
# STRIP THE EMAIL AND TOKEN!
JIRA_EMAIL = os.environ["JIRA_EMAIL"].strip()
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"].strip()

print(f"Testing cleaned credentials...")
print(f"Email length: {len(JIRA_EMAIL)}")
print(f"Token length: {len(JIRA_API_TOKEN)}")

r = requests.get(
    f"{JIRA_URL}/rest/api/3/myself",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json"}
)

print(f"Status: {r.status_code}")

if r.status_code == 200:
    data = r.json()
    print(f"SUCCESS! Logged in as: {data.get('displayName')}")
    print(f"Email: {data.get('emailAddress')}")
else:
    print(f"FAILED: {r.text}")
