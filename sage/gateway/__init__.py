"""SAGE Gateway — always-on daemon with HTTP gateway for cross-machine communication."""

from sage.gateway.message_queue import MessageQueue, PendingMessage
from sage.gateway.machine_config import SAGEMachineConfig, detect_machine
