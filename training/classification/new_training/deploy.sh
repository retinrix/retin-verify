#!/bin/bash
# Deploy v3 training to Colab - Interactive Script

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        Deploy v3 Training with Synthetic + Real Data              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if host is provided as argument
if [ -z "$1" ]; then
    echo "❌ Usage: ./deploy.sh <colab-host>"
    echo ""
    echo "Example:"
    echo "  ./deploy.sh abc123.trycloudflare.com"
    echo "  ./deploy.sh xxx.ngrok.io"
    echo ""
    echo "Get your host from the Colab tunnel output (cloudflared/ngrok)"
    exit 1
fi

HOST="$1"

echo "🎯 Colab Host: $HOST"
echo ""

# Verify SSH connection
echo "Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "root@$HOST" "echo 'Connected!'" 2>/dev/null; then
    echo "❌ Cannot connect to $HOST"
    echo ""
    echo "Make sure:"
    echo "  1. Colab is running with the tunnel active"
    echo "  2. The hostname is correct"
    echo "  3. SSH is configured (run setup_colab_ssh.py if needed)"
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Update the deploy script with the host
sed -i "s/HOST = \"YOUR_COLAB_HOST_HERE\"/HOST = \"$HOST\"/" deploy_v3_with_synthetic.py

echo "🚀 Starting deployment..."
python3 deploy_v3_with_synthetic.py

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Deployment initiated!"
echo ""
echo "To monitor training:"
echo "  ssh root@$HOST 'tail -f /content/retin_v3_synthetic/train_v3_synthetic.log'"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
