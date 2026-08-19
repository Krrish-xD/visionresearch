#!/bin/bash

echo "Starting Git commit sequence..."

# 1. Project Configuration
echo "Committing project config..."
git add .gitignore requirements.txt start.py test.sh RESEARCH_EXECUTION_PLAN.md 2>/dev/null
git commit -m "chore: initial project configuration and orchestration"

# 2. Backend Engine
echo "Committing backend engine..."
git add backend/ src/ 2>/dev/null
git commit -m "feat: implement visualizer backend engine and FastAPI routes"

# 3. Frontend Application
echo "Committing frontend..."
git add frontend/ 2>/dev/null
git commit -m "feat: scaffold React visualizer interface with multi-tab layout"

# 4. Testing Suite
echo "Committing tests..."
git add tests/ 2>/dev/null
git commit -m "test: add comprehensive pytest suite for VLM routing heuristics"

# Catch-all for any remaining loose files (like old mockups if any exist)
echo "Committing any remaining files..."
git add .
git commit -m "chore: cleanup and final file adjustments"

echo "Done! Your repository is cleanly staged and committed across 5 isolated commits."
