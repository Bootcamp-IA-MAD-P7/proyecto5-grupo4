import os
import requests

JIRA_URL = os.environ["JIRA_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

print(f"Secret Email: {JIRA_EMAIL}")

print("\n--- Asking Jira 'Who am I?' ---")
r = requests.get(
    f"{JIRA_URL}/rest/api/3/myself",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json"}
)

print(f"Status: {r.status_code}")
try:
    data = r.json()
    print(f"API Account Name: {data.get('displayName')}")
    print(f"API Account Email: {data.get('emailAddress')}")
except:
    print(f"Raw response: [{r.text}]")
