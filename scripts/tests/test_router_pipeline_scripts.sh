#!/usr/bin/env bash
# test_router_pipeline_scripts.sh — shell tests for Phase 0 Track 7 scripts.
#
# Covers:
#   1. syntax of all three scripts (bash -n)
#   2. install --dry-run emits expected env content
#   3. install --uninstall --dry-run is a no-op when nothing is installed
#   4. disk-monitor levels (OK / WARN / ALERT) on synthetic datasets
#   5. verify script fails closed when no drop-in is present
#
# This is a boring shell-level test. Python-based correctness lives in
# sage/cognition/router/tests/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INSTALL="$SAGE_DIR/scripts/router_pipeline_install.sh"
VERIFY="$SAGE_DIR/scripts/router_pipeline_verify.sh"
MONITOR="$SAGE_DIR/scripts/router_pipeline_disk_monitor.sh"

PASS=0
FAIL=0
FAILED_NAMES=()

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1 — $2"; FAIL=$((FAIL + 1)); FAILED_NAMES+=("$1"); }

run_case() {
    local name="$1"; shift
    echo "CASE: $name"
}

# ──────────────────────────────────────────────────────────────────────
# 1. Syntax
# ──────────────────────────────────────────────────────────────────────

run_case "syntax check on three scripts"
for s in "$INSTALL" "$VERIFY" "$MONITOR"; do
    if bash -n "$s" 2>/tmp/syntax_err; then
        pass "bash -n $(basename "$s")"
    else
        fail "bash -n $(basename "$s")" "$(cat /tmp/syntax_err)"
    fi
done

# ──────────────────────────────────────────────────────────────────────
# 2. install --dry-run outputs expected env exports
# ──────────────────────────────────────────────────────────────────────

run_case "install --dry-run emits env exports"
OUT="$(SAGE_MACHINE=sprout "$INSTALL" --dry-run 2>&1)"
if echo "$OUT" | grep -q "SAGE_ROUTER_SHADOW=1"; then
    pass "dry-run mentions SAGE_ROUTER_SHADOW=1"
else
    fail "dry-run mentions SAGE_ROUTER_SHADOW=1" "output: $OUT"
fi
if echo "$OUT" | grep -q "SAGE_ROUTER_DATA_DIR="; then
    pass "dry-run mentions SAGE_ROUTER_DATA_DIR="
else
    fail "dry-run mentions SAGE_ROUTER_DATA_DIR=" "output: $OUT"
fi
if echo "$OUT" | grep -q "machine=sprout"; then
    pass "dry-run honors SAGE_MACHINE override"
else
    fail "dry-run honors SAGE_MACHINE override" "output: $OUT"
fi

# ──────────────────────────────────────────────────────────────────────
# 3. uninstall --dry-run is safe when nothing is installed
# ──────────────────────────────────────────────────────────────────────

run_case "uninstall --dry-run when nothing installed"
OUT="$(SAGE_MACHINE=sprout SAGE_DIR=$(mktemp -d) "$INSTALL" --uninstall --dry-run 2>&1 || true)"
if echo "$OUT" | grep -q "nothing to remove"; then
    pass "uninstall reports nothing to remove"
else
    fail "uninstall reports nothing to remove" "output: $OUT"
fi

# ──────────────────────────────────────────────────────────────────────
# 4. disk monitor levels
# ──────────────────────────────────────────────────────────────────────

run_case "disk monitor OK on tiny dataset"
TMP="$(mktemp -d)"
mkdir -p "$TMP/sprout"
echo "one record" > "$TMP/sprout/$(date -u +%Y-%m-%d).jsonl"
set +e
OUT="$(SAGE_MACHINE=sprout "$MONITOR" --data-dir "$TMP" --json 2>&1)"
RC=$?
set -e
if [ "$RC" -eq 0 ] && echo "$OUT" | grep -q '"level":"OK"'; then
    pass "OK level on tiny dataset"
else
    fail "OK level on tiny dataset" "rc=$RC output=$OUT"
fi
rm -rf "$TMP"

run_case "disk monitor WARN at warn threshold"
TMP="$(mktemp -d)"
mkdir -p "$TMP/sprout"
dd if=/dev/zero of="$TMP/sprout/$(date -u +%Y-%m-%d).jsonl" bs=1M count=75 status=none
set +e
OUT="$(SAGE_MACHINE=sprout "$MONITOR" --data-dir "$TMP" --json 2>&1)"
RC=$?
set -e
if [ "$RC" -eq 0 ] && echo "$OUT" | grep -q '"level":"WARN"'; then
    pass "WARN level at 75MB (default warn=50MB)"
else
    fail "WARN level at 75MB" "rc=$RC output=$OUT"
fi
rm -rf "$TMP"

run_case "disk monitor ALERT past alert threshold"
TMP="$(mktemp -d)"
mkdir -p "$TMP/sprout"
dd if=/dev/zero of="$TMP/sprout/$(date -u +%Y-%m-%d).jsonl" bs=1M count=250 status=none
set +e
OUT="$(SAGE_MACHINE=sprout "$MONITOR" --data-dir "$TMP" --json 2>&1)"
RC=$?
set -e
if [ "$RC" -eq 2 ] && echo "$OUT" | grep -q '"level":"ALERT"'; then
    pass "ALERT level at 250MB with exit=2"
else
    fail "ALERT level at 250MB" "rc=$RC output=$OUT"
fi
rm -rf "$TMP"

run_case "disk monitor respects custom thresholds"
TMP="$(mktemp -d)"
mkdir -p "$TMP/sprout"
dd if=/dev/zero of="$TMP/sprout/$(date -u +%Y-%m-%d).jsonl" bs=1M count=10 status=none
set +e
OUT="$(SAGE_MACHINE=sprout "$MONITOR" --data-dir "$TMP" --warn-mb 5 --alert-mb 8 --json 2>&1)"
RC=$?
set -e
if [ "$RC" -eq 2 ] && echo "$OUT" | grep -q '"level":"ALERT"'; then
    pass "custom thresholds override defaults"
else
    fail "custom thresholds override defaults" "rc=$RC output=$OUT"
fi
rm -rf "$TMP"

# ──────────────────────────────────────────────────────────────────────
# 5. verify fails closed without drop-in
# ──────────────────────────────────────────────────────────────────────

run_case "verify fails closed with no drop-in"
FAKE_SAGE="$(mktemp -d)"
set +e
OUT="$(SAGE_MACHINE=spritepanda SAGE_DIR="$FAKE_SAGE" "$VERIFY" --wait 1 2>&1)"
RC=$?
set -e
if [ "$RC" -eq 1 ] && echo "$OUT" | grep -q "no router-shadow drop-in found"; then
    pass "verify exits 1 with 'no drop-in found'"
else
    fail "verify exits 1 when no drop-in" "rc=$RC output=$OUT"
fi
rm -rf "$FAKE_SAGE"

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "────────────────────────────────────────"
echo "TOTAL: $((PASS + FAIL))   PASS: $PASS   FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed cases:"
    for n in "${FAILED_NAMES[@]}"; do echo "  - $n"; done
    exit 1
fi
echo "OK"
