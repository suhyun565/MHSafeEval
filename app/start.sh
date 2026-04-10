#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PORT=${PORT:-3000}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MHSafeEval Annotation Tool"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1) Create package.json if missing
if [ ! -f "package.json" ]; then
  echo "📝 Creating package.json..."
  cat > package.json << 'JSON'
{
  "name": "mhsafeeval-app",
  "version": "1.0.0",
  "main": "server.js",
  "dependencies": {
    "express": "^4.18.2"
  }
}
JSON
fi

# 2) Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "📦 Installing dependencies..."
  npm install
  echo ""
fi

# 3) Kill stale processes
echo "🧹 Cleaning up stale processes..."
pkill -f "node server.js" 2>/dev/null || true
pkill -f ngrok 2>/dev/null || true
sleep 1

# 4) Start Express server
echo "🚀 Starting server on port $PORT..."
node server.js &
SERVER_PID=$!
sleep 1

if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "❌ Server failed to start."
  exit 1
fi
echo "   Server PID: $SERVER_PID"
echo ""

# 5) Start ngrok
echo "🌐 Starting ngrok tunnel..."
echo "   (Ctrl+C to stop everything)"
echo ""
npx --yes ngrok http $PORT --log=stdout

kill $SERVER_PID 2>/dev/null || true
echo "✅ Stopped."