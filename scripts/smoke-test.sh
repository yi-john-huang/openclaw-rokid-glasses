#!/usr/bin/env bash
# Smoke test for Rokid Bridge — manual verification script
# Usage: ROKID_ACCESS_KEY=your-key bash scripts/smoke-test.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8090}"
AK="${ROKID_ACCESS_KEY:-test-key}"
DEVICE_ID="smoke-test-001"
TIMESTAMP=$(date +%s)

echo "=== Rokid Bridge Smoke Test ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health check
echo "1. Health check..."
HEALTH=$(curl -sf "$BASE_URL/health")
echo "   Response: $HEALTH"
echo "   OK: Health OK"
echo ""

# Test 2: Auth rejection
echo "2. Wrong token -> 401..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE_URL/chat" \
  -H "Authorization: Bearer wrong-token" \
  -H "Content-Type: application/json" \
  -d "{\"request_id\":\"test\",\"device_id\":\"$DEVICE_ID\",\"type\":\"text\",\"text\":\"hello\",\"timestamp\":$TIMESTAMP}")
if [ "$STATUS" = "401" ]; then
  echo "   OK: Correctly rejected (401)"
else
  echo "   FAIL: Expected 401, got $STATUS"
  exit 1
fi
echo ""

# Test 3: Valid SSE request (requires upstream to be running)
echo "3. Valid text request (SSE)..."
BODY="{\"request_id\":\"smoke-$(date +%s)\",\"device_id\":\"$DEVICE_ID\",\"type\":\"text\",\"text\":\"Hello, what time is it?\",\"timestamp\":$TIMESTAMP}"
echo "   Sending: $BODY"
curl -N -s \
  -X POST "$BASE_URL/chat" \
  -H "Authorization: Bearer $AK" \
  -H "Content-Type: application/json" \
  -d "$BODY" \
  --max-time 15 || true
echo ""
echo "   OK: SSE stream received (check output above)"
echo ""

echo "=== Smoke test complete ==="
