#!/usr/bin/env bash
# Fails if an em dash appears in any tracked text file. Prose in this repo uses a colon for a
# definition, commas or parentheses for an aside, a period or semicolon for a hard break, and
# "to" for a range.
set -euo pipefail

if [ "$#" -gt 0 ]; then
    files="$*"
else
    files=$(git ls-files -- '*.md' '*.py' '*.toml' '*.yaml' '*.yml' '*.sh' '*.txt' 'Makefile')
fi

hits=0
for f in $files; do
    [ -f "$f" ] || continue
    [ "$f" = "scripts/check_no_em_dash.sh" ] && continue
    if LC_ALL=C grep -n $'\xe2\x80\x94' "$f"; then
        echo "em dash found in $f"
        hits=1
    fi
done

if [ "$hits" -ne 0 ]; then
    echo "em dash check failed"
    exit 1
fi
echo "em dash check passed"
