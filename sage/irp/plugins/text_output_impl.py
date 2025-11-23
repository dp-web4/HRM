#!/usr/bin/env python3
"""
TextOutput IRP Plugin

Provides real-time text output to both terminal and file for debugging/monitoring.
Launches its own terminal window showing live log updates.

Usage:
    plugin = TextOutputIRP({
        'log_file': '/tmp/sage_debug.log',
        'terminal': 'gnome-terminal',  # or 'xterm'
        'auto_launch': True
    })

    state = plugin.init_state({'text': 'Hello, world!'})
    state = plugin.step(state)
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class TextOutputIRP:
    """
    IRP-compliant plugin for debug/monitoring text output.

    Writes to both file and dedicated terminal window for real-time viewing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize text output stream."""
        self.config = config or {}

        # Configuration
        self.log_file = self.config.get('log_file', '/tmp/sage_debug.log')
        self.terminal = self.config.get('terminal', 'gnome-terminal')
        self.auto_launch = self.config.get('auto_launch', True)
        self.append_mode = self.config.get('append_mode', True)
        self.timestamp = self.config.get('timestamp', True)

        # Terminal process
        self.terminal_proc = None

        # Initialize log file
        mode = 'a' if self.append_mode else 'w'
        with open(self.log_file, mode) as f:
            if not self.append_mode:
                f.write(f"=== SAGE Debug Stream Started: {datetime.now()} ===\n\n")

        # Launch terminal if requested
        if self.auto_launch:
            self._launch_terminal()

        print(f"TextOutput IRP initialized: {self.log_file}")

    def _launch_terminal(self):
        """Launch terminal window showing live log."""
        try:
            if self.terminal == 'gnome-terminal':
                # GNOME Terminal
                cmd = [
                    'gnome-terminal',
                    '--',
                    'bash', '-c',
                    f'tail -f {self.log_file}; exec bash'
                ]
            elif self.terminal == 'xterm':
                # XTerm
                cmd = [
                    'xterm',
                    '-e',
                    f'tail -f {self.log_file}'
                ]
            elif self.terminal == 'konsole':
                # KDE Konsole
                cmd = [
                    'konsole',
                    '-e',
                    f'tail -f {self.log_file}'
                ]
            else:
                print(f"Unknown terminal: {self.terminal}, skipping launch")
                return

            self.terminal_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"✓ Launched {self.terminal} showing {self.log_file}")

        except FileNotFoundError:
            print(f"⚠ {self.terminal} not found, logging to file only")
        except Exception as e:
            print(f"⚠ Failed to launch terminal: {e}")

    def init_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Initialize output state.

        Args:
            context: Must contain 'text' key with message to output

        Returns:
            State dict for IRP iteration
        """
        text = context.get('text', '')
        prefix = context.get('prefix', '')
        suffix = context.get('suffix', '\n')

        state = {
            'text': text,
            'prefix': prefix,
            'suffix': suffix,
            'written': False,
            'iteration': 0,
            'energy': 1.0  # Will drop to 0 after writing
        }

        return state

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Execute one refinement step.

        For TextOutput, this is a single write operation.
        """
        if state['written']:
            # Already done
            state['energy'] = 0.0
            return state

        # Format message
        text = state['text']
        prefix = state['prefix']
        suffix = state['suffix']

        if self.timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            message = f"[{timestamp}] {prefix}{text}{suffix}"
        else:
            message = f"{prefix}{text}{suffix}"

        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(message)
                f.flush()  # Ensure immediate write for tail -f

            state['written'] = True
            state['energy'] = 0.0  # Done, no refinement needed

        except Exception as e:
            print(f"Error writing to {self.log_file}: {e}")
            state['energy'] = 0.5  # Partial failure

        state['iteration'] += 1
        return state

    def energy(self, state: Dict[str, Any]) -> float:
        """
        IRP Protocol: Return current energy.

        Low energy for I/O operations (no iterative refinement).
        """
        return state.get('energy', 1.0)

    def halt(self, state: Dict[str, Any]) -> bool:
        """
        IRP Protocol: Determine if processing should stop.

        Stop after successful write (energy = 0).
        """
        return state.get('written', False) or state['energy'] < 0.1

    def write(self, text: str, prefix: str = '', suffix: str = '\n'):
        """
        Convenience method for direct writing.

        Args:
            text: Message to write
            prefix: Optional prefix (e.g., "DEBUG: ")
            suffix: Optional suffix (default newline)
        """
        state = self.init_state({
            'text': text,
            'prefix': prefix,
            'suffix': suffix
        })

        while not self.halt(state):
            state = self.step(state)

        return state['written']

    def close(self):
        """Close terminal and cleanup."""
        if self.terminal_proc:
            self.terminal_proc.terminate()
            print(f"Closed terminal for {self.log_file}")


def test_text_output():
    """Test TextOutput IRP plugin."""
    print("="*80)
    print("Testing TextOutput IRP Plugin")
    print("="*80)
    print()

    # Initialize plugin (launches terminal)
    plugin = TextOutputIRP({
        'log_file': '/tmp/sage_test_output.log',
        'terminal': 'gnome-terminal',
        'auto_launch': True,
        'append_mode': False,
        'timestamp': True
    })

    # Test 1: Direct write
    print("Test 1: Direct write")
    plugin.write("Hello from SAGE!", prefix="INFO: ")

    # Test 2: IRP protocol
    print("\nTest 2: IRP protocol")
    context = {'text': 'Testing IRP iteration', 'prefix': 'DEBUG: '}
    state = plugin.init_state(context)

    while not plugin.halt(state):
        state = plugin.step(state)
        print(f"  Iteration {state['iteration']}, Energy: {plugin.energy(state):.3f}")

    # Test 3: Multiple messages
    print("\nTest 3: Multiple messages")
    for i in range(5):
        plugin.write(f"Message {i+1}", prefix="LOG: ")

    print("\n✓ Test complete - check the terminal window!")
    print(f"  Log file: {plugin.log_file}")

    input("\nPress Enter to close terminal and exit...")
    plugin.close()


if __name__ == "__main__":
    test_text_output()
