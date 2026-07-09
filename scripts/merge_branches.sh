#!/bin/bash
# scripts/merge_branches.sh
set -e

git remote add forensic-teammate <their-repo-url> 2>/dev/null || true
git fetch forensic-teammate
git subtree add --prefix=backend/agents/forensic forensic-teammate <branch-name> --squash

git remote add retrieval-teammate <their-repo-url> 2>/dev/null || true
git fetch retrieval-teammate
git subtree add --prefix=backend/agents/retrieval retrieval-teammate <branch-name> --squash

git remote add frontend-teammate <their-repo-url> 2>/dev/null || true
git fetch frontend-teammate
git subtree add --prefix=frontend frontend-teammate <branch-name> --squash

# to pull updates later, once someone pushes new work:
#   git subtree pull --prefix=backend/agents/forensic forensic-teammate <branch-name> --squash