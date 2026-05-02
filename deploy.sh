#!/bin/bash
# deploy.sh — Quick deploy helper for Vera Bot
# Supports: Railway, Render, Fly.io, local

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Vera Bot — Deployment Helper"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for API key
if [ -z "$GOOGLE_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠️  No LLM API key found."
  echo ""
  echo "Set one of:"
  echo "  export GOOGLE_API_KEY=your_key    ← Free at: https://aistudio.google.com/apikey"
  echo "  export ANTHROPIC_API_KEY=your_key ← https://console.anthropic.com"
  echo ""
  echo "Gemini Flash is FREE (no credit card needed) — recommended."
  echo ""
fi

MODE=${1:-local}

case "$MODE" in
  local)
    echo "→ Starting local server on :8080"
    pip install -r requirements.txt -q
    GOOGLE_API_KEY=$GOOGLE_API_KEY \
    ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    TEAM_NAME="${TEAM_NAME:-YourName}" \
    CONTACT_EMAIL="${CONTACT_EMAIL:-you@example.com}" \
    uvicorn bot:app --host 0.0.0.0 --port 8080 --log-level info
    ;;

  railway)
    echo "→ Deploying to Railway"
    echo ""
    echo "1. Install Railway CLI: npm install -g @railway/cli"
    echo "2. railway login"
    echo "3. railway init"
    echo "4. Set env vars:"
    echo "   railway vars set GOOGLE_API_KEY=xxx"
    echo "   railway vars set TEAM_NAME=YourName"
    echo "   railway vars set CONTACT_EMAIL=you@example.com"
    echo "5. railway up"
    echo ""
    echo "Your bot URL: https://<project>.railway.app"
    ;;

  render)
    echo "→ Deploying to Render"
    echo ""
    echo "1. Push this folder to a GitHub repo"
    echo "2. Go to https://dashboard.render.com/new/web"
    echo "3. Connect your repo"
    echo "4. Build command: pip install -r requirements.txt"
    echo "5. Start command: uvicorn bot:app --host 0.0.0.0 --port \$PORT"
    echo "6. Add env vars: GOOGLE_API_KEY, TEAM_NAME, CONTACT_EMAIL"
    echo ""
    echo "Your bot URL: https://<service>.onrender.com"
    ;;

  fly)
    echo "→ Deploying to Fly.io"
    echo ""
    echo "1. Install flyctl: curl -L https://fly.io/install.sh | sh"
    echo "2. flyctl auth login"
    echo "3. flyctl launch (accepts Dockerfile)"
    echo "4. flyctl secrets set GOOGLE_API_KEY=xxx TEAM_NAME=YourName"
    echo "5. flyctl deploy"
    ;;

  docker)
    echo "→ Building Docker image"
    docker build -t vera-bot .
    echo ""
    echo "→ Running container"
    docker run -p 8080:8080 \
      -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
      -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
      -e TEAM_NAME="${TEAM_NAME:-YourName}" \
      -e CONTACT_EMAIL="${CONTACT_EMAIL:-you@example.com}" \
      vera-bot
    ;;

  *)
    echo "Usage: ./deploy.sh [local|railway|render|fly|docker]"
    exit 1
    ;;
esac
