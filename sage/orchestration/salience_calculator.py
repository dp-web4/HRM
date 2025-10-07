#!/usr/bin/env python3
"""
SAGE Autonomous Attention System - Salience Calculator

Monitors SAGE training, orchestration, and federation activity to calculate
when Claude's strategic attention is needed.

Architecture:
- L-Level: This script monitors autonomously
- Salience: Calculates "interestingness" score (0.0 - 1.0)
- H-Level: Claude wakes up when threshold exceeded
"""

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


class SAGESalienceCalculator:
    """Calculate attention salience for SAGE development"""

    def __init__(self, base_dir: str = "/home/dp/ai-workspace/HRM"):
        self.base_dir = Path(base_dir)
        self.sage_dir = self.base_dir / "sage"
        self.orchestration_dir = self.sage_dir / "orchestration"
        self.training_dir = self.sage_dir / "training"
        self.economy_dir = self.sage_dir / "economy"

        # State file
        self.state_file = Path("/tmp/sage_monitor_state.json")
        self.last_attention_file = Path("/tmp/claude_last_attention_sage.txt")

    def load_state(self) -> Dict:
        """Load previous monitoring state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            "last_check": None,
            "training_cycles_seen": 0,
            "status_md_hash": None,
            "groot_errors": [],
            "last_activity": {}
        }

    def save_state(self, state: Dict):
        """Save monitoring state"""
        state["last_check"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def check_training_activity(self, state: Dict) -> Tuple[float, List[str]]:
        """
        Check for new training activity
        Returns: (salience_contribution, reasons)
        """
        salience = 0.0
        reasons = []

        # Check for training logs
        training_logs = list(self.training_dir.rglob("*.log")) if self.training_dir.exists() else []
        recent_logs = [
            log for log in training_logs
            if (datetime.now() - datetime.fromtimestamp(log.stat().st_mtime)) < timedelta(hours=24)
        ]

        if recent_logs:
            salience += 0.3
            reasons.append(f"Active training: {len(recent_logs)} recent log files")

        # Check for training errors or checkpoints
        checkpoint_dirs = list(self.training_dir.rglob("checkpoint-*")) if self.training_dir.exists() else []
        new_checkpoints = [
            cp for cp in checkpoint_dirs
            if (datetime.now() - datetime.fromtimestamp(cp.stat().st_mtime)) < timedelta(hours=6)
        ]

        if new_checkpoints:
            salience += 0.2
            reasons.append(f"New checkpoints: {len(new_checkpoints)} created recently")

        return salience, reasons

    def check_groot_integration(self, state: Dict) -> Tuple[float, List[str]]:
        """
        Check GR00T integration status
        Returns: (salience_contribution, reasons)
        """
        salience = 0.0
        reasons = []

        # Check real_groot_sage.py for errors or TODO markers
        groot_file = self.orchestration_dir / "real_groot_sage.py"
        if groot_file.exists():
            content = groot_file.read_text()

            # Check for error patterns
            error_indicators = [
                "AttributeError",
                "TODO",
                "FIXME",
                "process_backbone_inputs",  # Known issue
                "# Fix:",
                "# Need to"
            ]

            found_issues = [ind for ind in error_indicators if ind in content]
            if found_issues:
                salience += 0.25
                reasons.append(f"GR00T integration issues detected: {', '.join(found_issues[:3])}")

        # Check for GR00T model weights
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            groot_models = list(cache_dir.glob("*groot*"))
            if not groot_models:
                salience += 0.15
                reasons.append("GR00T model weights not found in cache")

        return salience, reasons

    def check_status_changes(self, state: Dict) -> Tuple[float, List[str]]:
        """
        Check for STATUS.md updates
        Returns: (salience_contribution, reasons)
        """
        salience = 0.0
        reasons = []

        status_file = self.sage_dir / "STATUS.md"
        if status_file.exists():
            content = status_file.read_text()
            current_hash = hash(content)

            if state.get("status_md_hash") is not None:
                if current_hash != state["status_md_hash"]:
                    salience += 0.2
                    reasons.append("STATUS.md updated - new development progress")

            state["status_md_hash"] = current_hash

            # Check for "IN PROGRESS" or warning markers
            if "⚠️" in content or "IN PROGRESS" in content:
                salience += 0.15
                reasons.append("Active work in progress flagged in STATUS.md")

        return salience, reasons

    def check_orchestration_agents(self, state: Dict) -> Tuple[float, List[str]]:
        """
        Check orchestration agent activity
        Returns: (salience_contribution, reasons)
        """
        salience = 0.0
        reasons = []

        agents_dir = self.orchestration_dir / "agents"
        if not agents_dir.exists():
            return salience, reasons

        # Count agents
        agent_files = list(agents_dir.rglob("*.py"))

        # Check for recent modifications
        recent_agents = [
            agent for agent in agent_files
            if (datetime.now() - datetime.fromtimestamp(agent.stat().st_mtime)) < timedelta(hours=12)
        ]

        if len(recent_agents) > 3:
            salience += 0.2
            reasons.append(f"High agent activity: {len(recent_agents)} agents modified recently")
        elif recent_agents:
            salience += 0.1
            reasons.append(f"Agent activity: {len(recent_agents)} agents modified")

        return salience, reasons

    def check_economy_compliance(self, state: Dict) -> Tuple[float, List[str]]:
        """
        Check economic compliance validator status
        Returns: (salience_contribution, reasons)
        """
        salience = 0.0
        reasons = []

        validator_file = self.economy_dir / "compliance_validator.py"
        if validator_file.exists():
            mtime = datetime.fromtimestamp(validator_file.stat().st_mtime)
            if (datetime.now() - mtime) < timedelta(hours=6):
                salience += 0.15
                reasons.append("Compliance validator recently updated")

        # Check for compliance reports
        reports = list(self.economy_dir.rglob("*REPORT*.md")) if self.economy_dir.exists() else []
        recent_reports = [
            r for r in reports
            if (datetime.now() - datetime.fromtimestamp(r.stat().st_mtime)) < timedelta(hours=24)
        ]

        if recent_reports:
            salience += 0.1
            reasons.append(f"New compliance reports: {len(recent_reports)}")

        return salience, reasons

    def check_attention_absence(self, state: Dict) -> Tuple[float, List[str]]:
        """
        Check how long since last attention
        Returns: (salience_contribution, reasons)
        """
        salience = 0.0
        reasons = []

        if self.last_attention_file.exists():
            try:
                last_attention_str = self.last_attention_file.read_text().strip()
                last_attention = datetime.fromisoformat(last_attention_str)
                hours_since = (datetime.now() - last_attention).total_seconds() / 3600

                if hours_since > 48:
                    salience += 0.3
                    reasons.append(f"Long attention absence: {hours_since:.1f} hours since last check")
                elif hours_since > 24:
                    salience += 0.2
                    reasons.append(f"Attention absence: {hours_since:.1f} hours since last check")
                elif hours_since > 12:
                    salience += 0.1
                    reasons.append(f"Moderate absence: {hours_since:.1f} hours since last check")

            except Exception as e:
                pass
        else:
            # No record of attention - might be first run
            salience += 0.1
            reasons.append("No attention timestamp found")

        return salience, reasons

    def calculate_salience(self) -> Dict:
        """
        Calculate overall salience score
        Returns: Dict with score, reasons, and recommendations
        """
        state = self.load_state()

        total_salience = 0.0
        all_reasons = []

        # Check all metrics
        metrics = [
            ("Training Activity", self.check_training_activity),
            ("GR00T Integration", self.check_groot_integration),
            ("Status Changes", self.check_status_changes),
            ("Orchestration Agents", self.check_orchestration_agents),
            ("Economy Compliance", self.check_economy_compliance),
            ("Attention Absence", self.check_attention_absence),
        ]

        breakdown = {}
        for metric_name, check_func in metrics:
            try:
                salience, reasons = check_func(state)
                total_salience += salience
                if reasons:
                    all_reasons.extend(reasons)
                    breakdown[metric_name] = salience
            except Exception as e:
                all_reasons.append(f"Error checking {metric_name}: {str(e)}")

        # Cap at 1.0
        total_salience = min(total_salience, 1.0)

        # Save state
        self.save_state(state)

        # Generate recommendations
        recommendations = []
        if total_salience >= 0.7:
            recommendations.append("High priority: Multiple systems need attention")
        if "GR00T" in str(all_reasons):
            recommendations.append("Check GR00T integration status - may have API issues")
        if "training" in str(all_reasons).lower():
            recommendations.append("Review training logs and checkpoints")
        if "compliance" in str(all_reasons).lower():
            recommendations.append("Run compliance validator on recent work")

        return {
            "timestamp": datetime.now().isoformat(),
            "salience_score": total_salience,
            "threshold": 0.5,
            "attention_needed": total_salience >= 0.5,
            "reasons": all_reasons,
            "breakdown": breakdown,
            "recommendations": recommendations or ["Continue monitoring"]
        }


def main():
    """Run salience calculation and print results"""
    calc = SAGESalienceCalculator()
    result = calc.calculate_salience()

    print(json.dumps(result, indent=2))

    # Exit code indicates if attention needed
    return 1 if result["attention_needed"] else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
