#!/usr/bin/env bash
# Print the highest GLIBC/GLIBCXX symbol version required by each libur*.so*.
# Usage: check_glibc.sh DIR [MAX_GLIBC]   (exit 1 if any lib needs glibc > MAX_GLIBC)
set -euo pipefail
DIR="${1:?usage: $0 DIR [MAX_GLIBC]}"
MAX="${2:-}"
rc=0
for f in "$DIR"/libur*.so*; do
  [[ -L "$f" ]] && continue
  v="$(objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1 || true)"
  x="$(objdump -T "$f" | grep -o 'GLIBCXX_[0-9.]*' | sed 's/GLIBCXX_//' | sort -V | tail -1 || true)"
  echo "$(basename "$f"): max GLIBC_${v:-none}, max GLIBCXX_${x:-none}"
  if [[ -n "$MAX" && -n "$v" && "$(printf '%s\n%s\n' "$v" "$MAX" | sort -V | tail -1)" != "$MAX" ]]; then
    echo "  ERROR: requires glibc $v > $MAX"; rc=1
  fi
done
exit $rc
