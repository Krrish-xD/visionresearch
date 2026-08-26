#!/bin/bash
echo "Running backend pytest... (output hidden, saving to test_results.log)"
if [ -f "venv/bin/pytest" ]; then
    venv/bin/pytest tests/ -v > test_results.log 2>&1
else
    pytest tests/ -v > test_results.log 2>&1
fi
echo "Backend tests finished."

echo "Checking frontend compilation..."
cd frontend && npm run build >> ../test_results.log 2>&1
if [ $? -eq 0 ]; then
    echo "Frontend compiled successfully."
else
    echo "Frontend compilation failed! See test_results.log for syntax errors."
fi

echo "All tests finished. Check test_results.log for full details."
