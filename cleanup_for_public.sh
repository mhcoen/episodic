#!/bin/bash
# Cleanup script to prepare Episodic repository for going public

echo "🧹 Cleaning up Episodic repository for public release..."

# Create archive directory if it doesn't exist
mkdir -p archive

# 1. Remove temporary/backup files
echo "Removing temporary files..."
rm -f "#PROJECT_MEMORY.md#"  # Emacs backup
rm -f profile.stats
rm -f profile_final.stats
rm -f analyze_profile.py
rm -f suppress_litellm_warning.py
rm -f topic_detection_output.txt
rm -f test_tab_completion.py  # This is a test file in root

# 2. Move session state files to archive
echo "Moving session state files to archive..."
[ -f "DOCUMENTATION_UPDATES.md" ] && mv DOCUMENTATION_UPDATES.md archive/
[ -f "SESSION_STATE_TAB_COMPLETION.md" ] && mv SESSION_STATE_TAB_COMPLETION.md archive/

# 3. Remove test outputs
echo "Removing test outputs..."
rm -rf tests/test_outputs/

# 4. Remove build artifacts
echo "Removing build artifacts..."
rm -rf episodic.egg-info/
rm -rf build/
rm -rf dist/
rm -rf .pytest_cache/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name ".DS_Store" -delete

# 5. Remove logs directory
echo "Removing logs..."
rm -rf logs/

# 6. Remove working directory
echo "Removing working directory..."
rm -rf working/

# 7. Check if private directory should be kept
if [ -d "private/" ]; then
    echo "⚠️  Found 'private/' directory - make sure it's in .gitignore!"
fi

# 8. Check exports directory
if [ -d "exports/" ]; then
    echo "⚠️  Found 'exports/' directory with example files - you may want to keep these as examples"
fi

# 9. Update .gitignore with any missing entries
echo "Updating .gitignore..."
# Check if these entries exist, if not add them
entries_to_add=(
    "private/"
    "logs/"
    "exports/"
    "*.stats"
    "profile_*.stats"
    ".venv/"
    "venv/"
    ".env"
    ".env.*"
    "*.key"
    "*.pem"
    "config.json"
    "*.tmp"
    "*.temp"
    "*.log"
    "*.bak"
    ".episodic/"
    "~/.episodic/"
    "working/"
    "*.swp"
    "*.swo"
    ".DS_Store"
)

for entry in "${entries_to_add[@]}"; do
    if ! grep -q "^${entry}$" .gitignore 2>/dev/null; then
        echo "$entry" >> .gitignore
        echo "  Added: $entry"
    fi
done

echo "✅ Updated .gitignore"

# 10. Check for sensitive data in files
echo "Checking for potential sensitive data..."
echo "⚠️  Please manually review these files for API keys or sensitive data:"
grep -l -i "api_key\|secret\|password\|token" --include="*.py" --include="*.md" --include="*.txt" -r . 2>/dev/null | grep -v ".venv" | grep -v "__pycache__" | grep -v ".git" | head -20

# 11. Final recommendations
echo ""
echo "📋 Final checklist before going public:"
echo "1. ✓ CLAUDE.md - Trimmed for public (full history in private/)"
echo "2. ✓ PROJECT_MEMORY.md - Trimmed for public (full history in private/)" 
echo "3. ✓ CURRENT_STATE.md - Trimmed for public (full history in private/)"
echo "4. ✓ Check git history for sensitive commits (API keys, etc.)"
echo "5. ✓ Ensure private/ is in .gitignore"
echo "6. ✓ Consider keeping exports/ as example files"
echo "7. ✓ Run 'git status' to see what will be committed"
echo ""
echo "🎉 Cleanup complete! Ready to make repository public."