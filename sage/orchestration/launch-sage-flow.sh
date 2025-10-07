#!/bin/bash

echo "🚀 Launching SAGE Multi-Agent System"
echo "===================================="

# Check dependencies
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    exit 1
fi

# Option menu
echo "Select operation mode:"
echo "1) Test agents"
echo "2) Start orchestrator"
echo "3) Launch with claude-flow swarm"
echo "4) Launch with hive-mind"
echo "5) Initialize only"

read -p "Choice [1-5]: " choice

case $choice in
    1)
        echo "🧪 Testing agents..."
        node test-agents.js
        ;;
    2)
        echo "🚀 Starting orchestrator..."
        node sage-orchestrator.js
        ;;
    3)
        echo "🐝 Launching with swarm..."
        npx claude-flow@alpha swarm "Build SAGE system" --claude
        ;;
    4)
        echo "🧠 Launching hive-mind..."
        npx claude-flow@alpha hive-mind wizard
        ;;
    5)
        echo "🔧 Initializing..."
        npx claude-flow@alpha init --force
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
