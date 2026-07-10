import os
import requests

JIRA_URL = os.environ["JIRA_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

print("=== PROJECT VERIFICATION ===\n")

# 1. Does 'ML' exist?
print("1. Checking if project 'ML' exists...")
r1 = requests.get(
    f"{JIRA_URL}/rest/api/3/project/ML",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json"}
)
print(f"   Status: {r1.status_code}")
if r1.status_code == 200:
    proj = r1.json()
    print(f"   Name: {proj.get('name')}")
    print(f"   Key: {proj.get('key')}")
else:
    print(f"   Response: {r1.text[:300]}")

# 2. List ALL projects
print("\n2. Listing ALL projects you have access to...")
r2 = requests.get(
    f"{JIRA_URL}/rest/api/3/project",
    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={"Accept": "application/json"}
)
print(f"   Status: {r2.status_code}")
if r2.status_code == 200:
    projects = r2.json()
    print(f"   Total projects: {len(projects)}")
    for p in projects:
        print(f"   -> Key: {p['key']}  |  Name: {p['name']}")
else:
    print(f"   Response: {r2.text[:300]}")

print("\n=== DONE ===")
