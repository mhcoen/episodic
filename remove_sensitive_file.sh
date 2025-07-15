#!/bin/bash

# Script to remove .mcp.json from git history

echo "WARNING: This will rewrite git history!"
echo "Make sure you have:"
echo "1. Revoked the exposed API key"
echo "2. Backed up your repository"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Option 1: Using git filter-branch (built-in)
echo "Removing .mcp.json from all history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch .mcp.json' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
echo "Cleaning up..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! Now you need to:"
echo "1. Force push to all remotes: git push --force --all"
echo "2. Force push tags: git push --force --tags"
echo "3. Tell all collaborators to re-clone the repository"