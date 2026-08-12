#!/usr/bin/env bash
# =============================================================================
# TabPFN Ablation Study — Verification Harness
#
# Single-command quality gate. Run after any code change:
#   bash scripts/verify.sh
#
# Steps 1-4 are pass/fail gates (exit non-zero on failure).
# Step 5 (coverage) is informational only.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PASS="${GREEN}✓ PASS${NC}"
FAIL="${RED}✗ FAIL${NC}"

# Track results
declare -a STEP_NAMES=()
declare -a STEP_RESULTS=()
OVERALL=0

run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local cmd=("$@")

    STEP_NAMES+=("$step_name")
    printf "${BLUE}[%s]${NC} ${BOLD}%s${NC} ... " "$step_num" "$step_name"

    output=$("${cmd[@]}" 2>&1)
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        STEP_RESULTS+=("pass")
        printf "%b\n" "$PASS"
    else
        STEP_RESULTS+=("fail")
        printf "%b\n" "$FAIL"
        echo ""
        echo "$output"
        echo ""
        OVERALL=1
    fi
}

run_test_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local cmd=("$@")

    STEP_NAMES+=("$step_name")
    printf "${BLUE}[%s]${NC} ${BOLD}%s${NC} ... \n" "$step_num" "$step_name"

    "${cmd[@]}"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        STEP_RESULTS+=("pass")
        printf "%b\n" "$PASS"
    else
        STEP_RESULTS+=("fail")
        printf "%b\n" "$FAIL"
        OVERALL=1
    fi
    echo ""
}

# =============================================================================
echo ""
echo -e "${BOLD}=== TabPFN Ablation — Verification Harness ===${NC}"
echo ""

# Step 1: Ruff lint
run_step "1/4" "Ruff lint" uv run ruff check .

# Step 2: Ruff format
run_step "2/4" "Ruff format check" uv run ruff format --check .

# Step 3: Pyright
run_step "3/4" "Pyright type check" uv run pyright

# Step 4: Pytest & Coverage
run_test_step "4/4" "Pytest and Coverage" uv run pytest tests/ --cov=. --cov-report=term-missing -q --no-header --override-ini="addopts="

# =============================================================================
# Summary
# =============================================================================
echo -e "${BOLD}=== Summary ===${NC}"
echo ""

for i in "${!STEP_NAMES[@]}"; do
    result="${STEP_RESULTS[$i]}"
    name="${STEP_NAMES[$i]}"
    if [ "$result" = "pass" ]; then
        printf "  %b  %s\n" "$PASS" "$name"
    elif [ "$result" = "fail" ]; then
        printf "  %b  %s\n" "$FAIL" "$name"
    else
        printf "  ${YELLOW}ℹ INFO${NC}  %s\n" "$name"
    fi
done

echo ""
if [ $OVERALL -eq 0 ]; then
    echo -e "${GREEN}${BOLD}=== ALL CHECKS PASSED ===${NC}"
else
    echo -e "${RED}${BOLD}=== SOME CHECKS FAILED ===${NC}"
fi
echo ""

exit $OVERALL
