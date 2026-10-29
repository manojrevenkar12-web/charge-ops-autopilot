#!/usr/bin/env bash
# Run a CI step; on failure, put the tail of its output in an annotation (logs are not always readable).
name="$1"; shift
out=$(mktemp)
if "$@" >"$out" 2>&1; then cat "$out"; exit 0; fi
status=$?
cat "$out"
msg=$(tail -c 3000 "$out" | sed -e 's/%/%25/g' -e 's/\r/%0D/g' | sed ':a;N;$!ba;s/\n/%0A/g')
echo "::error title=${name}::${msg}"
exit $status
