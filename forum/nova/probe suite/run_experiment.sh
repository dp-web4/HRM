#!/bin/bash
echo "Starting probe suite..."
python simulate_agent_stub.py --experiment 1 --trials 30
python simulate_agent_stub.py --experiment 2 --trials 50
python simulate_agent_stub.py --experiment 3 --trials 20
echo "Experiments complete. Logs stored in ./logs"
