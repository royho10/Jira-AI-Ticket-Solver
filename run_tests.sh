#!/usr/bin/env bash
#
# Test runner for Jira AI Ticket Solver
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh deterministic # Run only free/fast deterministic tests (no LLM calls)
#   ./run_tests.sh llm           # Run LLM evaluation tests (costs money)
#   ./run_tests.sh integration   # Run integration tests (costs money, needs LLM)
#   ./run_tests.sh all           # Run everything
#   ./run_tests.sh quick         # Alias for deterministic
#
set -euo pipefail

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

TIER="${1:-all}"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN} Jira AI Ticket Solver - Test Runner ${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

case "$TIER" in
  deterministic|quick)
    echo -e "${YELLOW}Running DETERMINISTIC tests only (no LLM calls, free)...${NC}"
    echo ""
    python -m pytest -m deterministic -v --tb=short "$@"
    ;;
  llm|llm_eval)
    echo -e "${YELLOW}Running LLM EVAL tests (requires Azure OpenAI, costs money)...${NC}"
    echo ""
    python -m pytest -m llm_eval -v --tb=short "$@"
    ;;
  integration)
    echo -e "${YELLOW}Running INTEGRATION tests (requires Azure OpenAI, costs money)...${NC}"
    echo ""
    python -m pytest -m integration -v --tb=short "$@"
    ;;
  all|"")
    echo -e "${YELLOW}Running ALL tests...${NC}"
    echo ""
    echo -e "${GREEN}--- Tier 1: Deterministic tests ---${NC}"
    python -m pytest -m deterministic -v --tb=short || true
    echo ""
    echo -e "${GREEN}--- Tier 2: LLM eval tests ---${NC}"
    python -m pytest -m llm_eval -v --tb=short || true
    echo ""
    echo -e "${GREEN}--- Tier 3: Integration tests ---${NC}"
    python -m pytest -m integration -v --tb=short || true
    ;;
  *)
    echo -e "${RED}Unknown tier: $TIER${NC}"
    echo ""
    echo "Usage: $0 [deterministic|quick|llm|integration|all]"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
