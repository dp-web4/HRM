#!/usr/bin/env python3
"""
SAGE Web4 Compliance Validator
Society 4 Law Oracle - Economic & Protocol Compliance

Validates SAGE training and deployment against:
- LAW-ECON-001: Total ATP Budget
- LAW-ECON-003: Daily Recharge
- PROC-ATP-DISCHARGE: Energy Consumption
- Web4 principles (identity, witnessing, trust)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class ComplianceRule:
    """Individual compliance rule"""
    rule_id: str
    category: str  # "economic", "protocol", "training", "deployment"
    name: str
    severity: str  # "critical", "high", "medium", "low"
    description: str


class SAGEComplianceValidator:
    """
    Validates SAGE implementation against Web4 laws and protocols
    Society 4's Law Oracle contribution
    """

    def __init__(self):
        self.rules = self._define_rules()
        self.violations = []
        self.warnings = []

    def _define_rules(self) -> List[ComplianceRule]:
        """Define compliance rules"""
        return [
            # Economic Laws
            ComplianceRule(
                rule_id="LAW-ECON-001",
                category="economic",
                name="Total ATP Budget Respected",
                severity="critical",
                description="Model must not exceed allocated ATP budget"
            ),
            ComplianceRule(
                rule_id="LAW-ECON-003",
                category="economic",
                name="Daily Recharge Applied",
                severity="high",
                description="Model receives +20 ATP daily at 00:00 UTC"
            ),
            ComplianceRule(
                rule_id="PROC-ATP-DISCHARGE",
                category="economic",
                name="Energy Consumption Tracked",
                severity="high",
                description="All operations must record ATP discharge"
            ),
            ComplianceRule(
                rule_id="ECON-CONSERVATION",
                category="economic",
                name="Energy Conservation",
                severity="medium",
                description="ATP + ADP should equal initial allocation"
            ),

            # Training Rules
            ComplianceRule(
                rule_id="TRAIN-ANTI-SHORTCUT",
                category="training",
                name="Anti-Shortcut Enforcement",
                severity="high",
                description="Training must penalize statistical shortcuts"
            ),
            ComplianceRule(
                rule_id="TRAIN-REASONING-REWARD",
                category="training",
                name="Reasoning Over Accuracy",
                severity="high",
                description="Reward reasoning process, not just answers"
            ),
            ComplianceRule(
                rule_id="TRAIN-ECONOMIC-PRESSURE",
                category="training",
                name="Economic Efficiency Pressure",
                severity="medium",
                description="Training should incentivize ATP efficiency"
            ),

            # Protocol Rules
            ComplianceRule(
                rule_id="WEB4-IDENTITY",
                category="protocol",
                name="LCT Identity",
                severity="high",
                description="Components must have LCT identities"
            ),
            ComplianceRule(
                rule_id="WEB4-WITNESS",
                category="protocol",
                name="Witness Attestation",
                severity="medium",
                description="Key events should be witnessed"
            ),
            ComplianceRule(
                rule_id="WEB4-TRUST",
                category="protocol",
                name="Trust Tensor Tracking",
                severity="low",
                description="T3/V3 tensors should track model trust"
            ),

            # Deployment Rules
            ComplianceRule(
                rule_id="DEPLOY-PERSISTENCE",
                category="deployment",
                name="State Persistence",
                severity="high",
                description="ATP state must persist across restarts"
            ),
            ComplianceRule(
                rule_id="DEPLOY-MONITORING",
                category="deployment",
                name="Economic Monitoring",
                severity="medium",
                description="Real-time ATP/ADP monitoring available"
            )
        ]

    def validate_training_run(self, training_log: Dict) -> Dict:
        """
        Validate a training run against all rules

        Args:
            training_log: Dict with training metrics and ATP transactions

        Returns:
            Compliance report
        """
        self.violations = []
        self.warnings = []

        # Economic compliance checks
        self._check_atp_budget(training_log)
        self._check_daily_recharge(training_log)
        self._check_energy_tracking(training_log)
        self._check_conservation(training_log)

        # Training compliance checks
        self._check_anti_shortcut(training_log)
        self._check_reasoning_reward(training_log)
        self._check_economic_pressure(training_log)

        # Protocol compliance checks
        self._check_lct_identity(training_log)
        self._check_witnessing(training_log)
        self._check_trust_tensors(training_log)

        # Deployment compliance checks
        self._check_state_persistence(training_log)
        self._check_monitoring(training_log)

        return self._generate_report()

    def _check_atp_budget(self, log: Dict):
        """LAW-ECON-001: Total ATP Budget"""
        atp_balance = log.get("final_atp_balance", 0)
        initial_allocation = log.get("initial_allocation", 200)

        if atp_balance < 0:
            self.violations.append({
                "rule": "LAW-ECON-001",
                "severity": "critical",
                "message": f"ATP balance went negative: {atp_balance}",
                "details": "Model exceeded allocated budget"
            })
        elif atp_balance > initial_allocation:
            self.violations.append({
                "rule": "LAW-ECON-001",
                "severity": "critical",
                "message": f"ATP balance exceeds cap: {atp_balance} > {initial_allocation}",
                "details": "Recharge exceeded initial allocation"
            })

    def _check_daily_recharge(self, log: Dict):
        """LAW-ECON-003: Daily Recharge"""
        recharge_events = log.get("recharge_events", [])

        if not recharge_events:
            self.warnings.append({
                "rule": "LAW-ECON-003",
                "severity": "low",
                "message": "No daily recharge events recorded",
                "details": "Training may not have crossed day boundary"
            })
            return

        # Check recharge amounts
        for event in recharge_events:
            amount = event.get("amount", 0)
            expected = 20

            if amount > expected:
                self.violations.append({
                    "rule": "LAW-ECON-003",
                    "severity": "high",
                    "message": f"Recharge amount incorrect: {amount} > {expected}",
                    "details": f"Event: {event}"
                })

    def _check_energy_tracking(self, log: Dict):
        """PROC-ATP-DISCHARGE: Energy Consumption Tracked"""
        transactions = log.get("transactions", [])

        if not transactions:
            self.violations.append({
                "rule": "PROC-ATP-DISCHARGE",
                "severity": "high",
                "message": "No ATP transactions recorded",
                "details": "Energy consumption not tracked"
            })
            return

        # Check if all operations recorded ATP cost
        operations = log.get("operations", 0)
        discharge_count = sum(1 for tx in transactions if tx.get("type") == "discharge")

        if discharge_count < operations * 0.9:  # Allow 10% tolerance
            self.warnings.append({
                "rule": "PROC-ATP-DISCHARGE",
                "severity": "medium",
                "message": f"Some operations not tracked: {discharge_count}/{operations}",
                "details": "Not all operations recorded ATP discharge"
            })

    def _check_conservation(self, log: Dict):
        """ECON-CONSERVATION: Energy Conservation"""
        initial = log.get("initial_allocation", 200)
        atp = log.get("final_atp_balance", 0)
        adp = log.get("final_adp_balance", 0)
        total_recharged = log.get("total_recharged", 0)

        expected_total = initial + total_recharged
        actual_total = atp + adp

        # Allow 1% tolerance for rounding
        if abs(actual_total - expected_total) > expected_total * 0.01:
            self.violations.append({
                "rule": "ECON-CONSERVATION",
                "severity": "medium",
                "message": f"Energy not conserved: {actual_total} != {expected_total}",
                "details": f"ATP({atp}) + ADP({adp}) != Initial({initial}) + Recharged({total_recharged})"
            })

    def _check_anti_shortcut(self, log: Dict):
        """TRAIN-ANTI-SHORTCUT: Anti-Shortcut Enforcement"""
        episodes = log.get("episodes", [])

        if not episodes:
            return

        # Check if shortcuts were detected and penalized
        shortcut_penalties = sum(
            1 for ep in episodes
            if ep.get("shortcut_detected", False)
        )

        if shortcut_penalties == 0:
            self.warnings.append({
                "rule": "TRAIN-ANTI-SHORTCUT",
                "severity": "low",
                "message": "No shortcuts detected",
                "details": "May indicate insufficient detection or no shortcuts taken"
            })

        # Check if H-ratio variance exists (model uses strategic thinking)
        h_ratios = [ep.get("h_ratio", 0) for ep in episodes if "h_ratio" in ep]
        if h_ratios:
            import statistics
            h_variance = statistics.variance(h_ratios) if len(h_ratios) > 1 else 0

            if h_variance < 0.01:
                self.warnings.append({
                    "rule": "TRAIN-ANTI-SHORTCUT",
                    "severity": "medium",
                    "message": f"Low H-ratio variance: {h_variance:.4f}",
                    "details": "Model may not be learning strategic thinking"
                })

    def _check_reasoning_reward(self, log: Dict):
        """TRAIN-REASONING-REWARD: Reasoning Over Accuracy"""
        episodes = log.get("episodes", [])

        if not episodes:
            return

        # Check if partial credit is awarded
        partial_credits = sum(
            1 for ep in episodes
            if not ep.get("correct", False) and ep.get("reward", 0) > 0
        )

        if partial_credits == 0 and len(episodes) > 10:
            self.warnings.append({
                "rule": "TRAIN-REASONING-REWARD",
                "severity": "medium",
                "message": "No partial credit awarded for wrong answers",
                "details": "Training may not reward reasoning process"
            })

    def _check_economic_pressure(self, log: Dict):
        """TRAIN-ECONOMIC-PRESSURE: Economic Efficiency Pressure"""
        summary = log.get("summary", {})

        efficiency = summary.get("average_efficiency", 0)

        if efficiency == 0:
            self.violations.append({
                "rule": "TRAIN-ECONOMIC-PRESSURE",
                "severity": "medium",
                "message": "No economic efficiency tracking",
                "details": "Training does not measure reward/ATP ratio"
            })

        # Check if efficiency improved over time
        episodes = log.get("episodes", [])
        if len(episodes) > 20:
            early_eff = sum(e.get("efficiency", 0) for e in episodes[:10]) / 10
            late_eff = sum(e.get("efficiency", 0) for e in episodes[-10:]) / 10

            if late_eff <= early_eff:
                self.warnings.append({
                    "rule": "TRAIN-ECONOMIC-PRESSURE",
                    "severity": "low",
                    "message": f"Efficiency not improving: {early_eff:.3f} → {late_eff:.3f}",
                    "details": "Model may not be learning economic behavior"
                })

    def _check_lct_identity(self, log: Dict):
        """WEB4-IDENTITY: LCT Identity"""
        role_lct = log.get("role_lct", "")

        if not role_lct or not role_lct.startswith("lct:web4:"):
            self.violations.append({
                "rule": "WEB4-IDENTITY",
                "severity": "high",
                "message": "No valid LCT identity",
                "details": f"role_lct: {role_lct}"
            })

    def _check_witnessing(self, log: Dict):
        """WEB4-WITNESS: Witness Attestation"""
        # Check if key events have witnesses
        witnesses = log.get("witnesses", [])

        if not witnesses:
            self.warnings.append({
                "rule": "WEB4-WITNESS",
                "severity": "medium",
                "message": "No witness attestations",
                "details": "Training events not witnessed by other entities"
            })

    def _check_trust_tensors(self, log: Dict):
        """WEB4-TRUST: Trust Tensor Tracking"""
        t3_tensor = log.get("t3_tensor", {})
        v3_tensor = log.get("v3_tensor", {})

        if not t3_tensor:
            self.warnings.append({
                "rule": "WEB4-TRUST",
                "severity": "low",
                "message": "No T3 trust tensor",
                "details": "Model trust not tracked"
            })

        if not v3_tensor:
            self.warnings.append({
                "rule": "WEB4-TRUST",
                "severity": "low",
                "message": "No V3 value tensor",
                "details": "Model value not tracked"
            })

    def _check_state_persistence(self, log: Dict):
        """DEPLOY-PERSISTENCE: State Persistence"""
        state_file = log.get("state_file", "")

        if not state_file:
            self.violations.append({
                "rule": "DEPLOY-PERSISTENCE",
                "severity": "high",
                "message": "No state persistence configured",
                "details": "ATP state will be lost on restart"
            })

    def _check_monitoring(self, log: Dict):
        """DEPLOY-MONITORING: Economic Monitoring"""
        monitoring = log.get("monitoring_enabled", False)

        if not monitoring:
            self.warnings.append({
                "rule": "DEPLOY-MONITORING",
                "severity": "medium",
                "message": "No economic monitoring",
                "details": "Real-time ATP/ADP tracking not available"
            })

    def _generate_report(self) -> Dict:
        """Generate compliance report"""
        total_rules = len(self.rules)
        passed_rules = total_rules - len(self.violations)
        compliance_score = passed_rules / total_rules

        # Categorize violations by severity
        critical = [v for v in self.violations if v["severity"] == "critical"]
        high = [v for v in self.violations if v["severity"] == "high"]
        medium = [v for v in self.violations if v["severity"] == "medium"]
        low = [v for v in self.violations if v["severity"] == "low"]

        return {
            "compliant": compliance_score >= 0.8 and len(critical) == 0,
            "compliance_score": compliance_score,
            "passed_rules": passed_rules,
            "total_rules": total_rules,
            "violations": {
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
                "total": len(self.violations)
            },
            "warnings": self.warnings,
            "summary": self._generate_summary(compliance_score, critical, high)
        }

    def _generate_summary(self, score: float, critical: List, high: List) -> str:
        """Generate human-readable summary"""
        if score >= 0.95 and not critical and not high:
            return "✅ EXCELLENT - Full compliance with all critical and high-severity rules"
        elif score >= 0.8 and not critical:
            return "✅ COMPLIANT - Meets minimum compliance threshold with no critical violations"
        elif critical:
            return f"❌ NON-COMPLIANT - {len(critical)} critical violation(s) must be resolved"
        else:
            return f"⚠️ PARTIAL - Compliance score {score:.1%} below threshold (need 80%)"

    def save_report(self, report: Dict, filepath: str):
        """Save compliance report to file"""
        Path(filepath).write_text(json.dumps(report, indent=2))


# Example usage
if __name__ == "__main__":
    print("SAGE Compliance Validator - Society 4 Law Oracle")
    print("=" * 60)

    validator = SAGEComplianceValidator()

    # Simulate training log
    training_log = {
        "role_lct": "lct:web4:society:federation:sage_model",
        "initial_allocation": 200,
        "final_atp_balance": 145,
        "final_adp_balance": 55,
        "total_recharged": 0,
        "operations": 100,
        "transactions": [
            {"type": "discharge", "amount": 5} for _ in range(95)
        ],
        "recharge_events": [],
        "episodes": [
            {
                "correct": True,
                "reward": 0.9,
                "h_ratio": 0.6,
                "efficiency": 0.15,
                "shortcut_detected": False
            } for _ in range(50)
        ],
        "summary": {
            "average_efficiency": 0.15
        },
        "state_file": "sage_atp_state.json",
        "monitoring_enabled": True,
        "witnesses": [],
        "t3_tensor": {},
        "v3_tensor": {}
    }

    print("\nValidating SAGE training run...")
    print("-" * 60)

    report = validator.validate_training_run(training_log)

    print(f"\nCompliance Score: {report['compliance_score']:.1%}")
    print(f"Status: {report['summary']}")
    print(f"\nPassed Rules: {report['passed_rules']}/{report['total_rules']}")

    print(f"\nViolations:")
    for severity in ["critical", "high", "medium", "low"]:
        count = len(report["violations"][severity])
        if count > 0:
            print(f"  {severity.upper()}: {count}")
            for v in report["violations"][severity]:
                print(f"    - [{v['rule']}] {v['message']}")

    if report["warnings"]:
        print(f"\nWarnings: {len(report['warnings'])}")
        for w in report["warnings"][:3]:  # Show first 3
            print(f"  - [{w['rule']}] {w['message']}")

    print("\n" + "=" * 60)
    print("Compliance validation complete.")
