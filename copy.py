import subprocess

REPO_URL = "git@github.com:YOUR_USERNAME/YOUR_REPOSITORY_NAME.git"

subprocess.run(["git", "clone", REPO_URL], check=True)
