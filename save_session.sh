#!/bin/bash
# Convenience script to save session state
# Usage: ./save_session.sh "Task Name" "status" "progress" "next_steps"

if [ $# -lt 4 ]; then
    echo "Usage: ./save_session.sh \"Task Name\" \"status\" \"progress\" \"next_steps\""
    echo ""
    echo "Example:"
    echo '  ./save_session.sh "v3 Training" "in_progress" "Epoch 25/50" "Complete training"'
    exit 1
fi

python3 .kimi/session_manager.py save "$1" "$2" "$3" "$4"
