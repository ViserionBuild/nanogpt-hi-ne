#!/usr/bin/env bash
# =============================================================================
# Package this repo for Kaggle.
#
# Produces two archives, because they have very different sizes and very
# different update frequencies:
#
#   dist/lma-code.zip   ~4.4 MB   code, configs, tokenizers, metadata.
#                                 Changes constantly. Re-upload freely.
#   dist/lma-data-*.zip ~2.6 GB   the tokenized corpora (.bin).
#                                 Written once in Phase 1. Upload ONCE.
#
# Keeping them separate is the whole point: a one-line code fix should not
# mean re-uploading gigabytes.
#
# Usage:
#   ./package_for_kaggle.sh           # both archives
#   ./package_for_kaggle.sh code      # code only (the common case)
#   ./package_for_kaggle.sh data      # data only
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
DIST="$REPO/dist"
WHAT="${1:-all}"

mkdir -p "$DIST"
cd "$REPO"

human() { du -h "$1" | cut -f1; }

# -----------------------------------------------------------------------------
# 1. Code archive — exactly what git tracks, minus the data directories.
#
# Using `git ls-files` rather than a hand-written include list means the
# archive can never drift from the repo: anything committed is included,
# anything gitignored (checkpoints, .bin, raw corpora) is not.
# -----------------------------------------------------------------------------
if [[ "$WHAT" == "all" || "$WHAT" == "code" ]]; then
  echo "==> building code archive"
  rm -f "$DIST/lma-code.zip"

  git ls-files \
    | grep -vE '^(hindi|nepali)/data/' \
    | grep -vE '\.(pdf|DS_Store)$' \
    > "$DIST/.codefiles.txt"

  zip -q -X "$DIST/lma-code.zip" -@ < "$DIST/.codefiles.txt"
  rm -f "$DIST/.codefiles.txt"

  echo "    dist/lma-code.zip  $(human "$DIST/lma-code.zip")  \
($(unzip -l "$DIST/lma-code.zip" | tail -1 | awk '{print $2}') files)"
fi

# -----------------------------------------------------------------------------
# 2. Data archives — one per language, so a re-upload of one language does
#    not disturb the other.
#
# -1 (fastest compression) is deliberate: uint32 token ids compress ~52%
# at level 1, and higher levels buy almost nothing on this data while
# taking many times longer.
# -----------------------------------------------------------------------------
if [[ "$WHAT" == "all" || "$WHAT" == "data" ]]; then
  for LANG in hindi nepali; do
    BINDIR="$LANG/data/tokenized"
    if ! compgen -G "$BINDIR/*.bin" > /dev/null; then
      echo "==> skipping $LANG (no .bin files in $BINDIR)"
      continue
    fi

    echo "==> building $LANG data archive (this takes a few minutes)"
    OUT="$DIST/lma-data-$LANG.zip"
    rm -f "$OUT"

    # Stored paths stay repo-relative so extraction on Kaggle reproduces
    # the same layout the configs expect.
    #
    # The tokenizer .model/.vocab ride along even though they are also in
    # lma-code.zip: under 1 MB against 1.3 GB, and it makes each data
    # dataset self-contained rather than silently depending on the code
    # dataset being attached too.
    zip -q -1 -X "$OUT" \
      "$BINDIR"/train.bin \
      "$BINDIR"/validation.bin \
      "$BINDIR"/test.bin \
      "$BINDIR"/tokenization_metadata.json \
      "$LANG"/tokenizer/model/"$LANG"_bpe_12000.model \
      "$LANG"/tokenizer/vocab/"$LANG"_bpe_12000.vocab

    echo "    $(basename "$OUT")  $(human "$OUT")"
  done
fi

echo
echo "Done. Archives in dist/:"
ls -lh "$DIST"/*.zip 2>/dev/null | awk '{printf "  %-28s %s\n", $9, $5}'
echo
echo "Next: upload dist/*.zip at https://www.kaggle.com/datasets  (New Dataset)"
echo "See KAGGLE_SETUP.md for the notebook."
