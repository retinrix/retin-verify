#!/bin/bash
# Start CNIE Classification Server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CNIE Classification Server           ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install fastapi uvicorn python-multipart -q
}

echo ""
echo -e "${GREEN}✓ Starting server...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create necessary directories
mkdir -p feedback_data/{misclassified,correct,low_confidence}
mkdir -p frontend/css frontend/js

# Start server
cd backend
python3 api_server.py "$@"
