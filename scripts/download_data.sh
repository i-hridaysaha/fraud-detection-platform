#!/usr/bin/env bash
# Downloads the two labelled training files of the Kaggle competition "ieee-fraud-detection"
# into data/. The data is Vesta-provided under the competition rules and is not redistributable,
# so the repo ships this script and never the CSVs.
#
# Requires Kaggle credentials (~/.kaggle/kaggle.json or ~/.kaggle/credentials.json) and acceptance
# of the competition rules on the Kaggle site. Neither is done for you here.
#
# The test files are deliberately not downloaded: they are unlabelled, so they cannot be used for
# training, evaluation, or the audit.
set -euo pipefail

COMP="ieee-fraud-detection"
DEST="data"
KAGGLE="${KAGGLE:-.venv/bin/kaggle}"
FILES=("train_transaction.csv" "train_identity.csv")

mkdir -p "$DEST"

echo "free space before:"
df -h "$DEST"

for f in "${FILES[@]}"; do
    if [ -f "$DEST/$f" ]; then
        echo "$f already present, skipping"
        continue
    fi
    echo "downloading $f"
    "$KAGGLE" competitions download -c "$COMP" -f "$f" -p "$DEST"
    if [ -f "$DEST/$f.zip" ]; then
        unzip -o -q "$DEST/$f.zip" -d "$DEST"
        rm -f "$DEST/$f.zip"
    fi
done

echo "free space after:"
df -h "$DEST"

echo "downloaded files:"
ls -l "$DEST"

echo "sha256:"
shasum -a 256 "$DEST"/*.csv
