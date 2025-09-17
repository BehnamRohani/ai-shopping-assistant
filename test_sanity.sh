#!/usr/bin/env bash
set -e
URL=${1:-http://localhost:8000/chat}
echo "Ping test..."
curl -s -X POST $URL -H "Content-Type: application/json" -d '{"chat_id":"sanity-check-ping","messages":[{"type":"text","content":"ping"}]}' | jq
echo -e "\nBase key test..."
curl -s -X POST $URL -H "Content-Type: application/json" -d '{"chat_id":"sanity-check-base-key","messages":[{"type":"text","content":"return base random key: 123e4567-e89b-12d3-a456-426614174000"}]}' | jq
