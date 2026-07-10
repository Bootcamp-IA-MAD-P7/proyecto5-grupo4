import os
import requests
import json

JIRA_URL = os.environ["JIRA_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

print("=== RAW RESPONSE TEST ===\n")

r = requests.post(
    f"{JIRA_URL}/rest/api/3/search/jql",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    json={"jql": "project = ML", "maxResults": 1, "fields": ["summary"]}
)

print(f"Status: {r.status_code}")
print("Raw JSON response:")
print(r.text)
