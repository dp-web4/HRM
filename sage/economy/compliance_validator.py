#!/usr/bin/env python3
"""
SAGE Web4 Compliance Validator v2.0
Society 4 Law Oracle - Economic & Protocol Compliance

Implements RFC-LAW-ALIGN-001 (Alignment vs Compliance) and RFC-R6-TO-R7-EVOLUTION (Explicit Reputation)

Validates SAGE training and deployment against:
- LAW-ECON-001: Total ATP Budget
- LAW-ECON-003: Daily Recharge
- PROC-ATP-DISCHARGE: Energy Consumption
- Web4 principles (identity, witnessing, trust)

New in v2.0:
- Alignment (spirit) vs Compliance (letter) distinction
- R7 framework with explicit ReputationDelta output
- Context-conditional compliance based on Web4 abstraction level
- Trust-building visibility
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path


class Verdict(Enum):
    """Validation verdict"""
    PERFECT = "PERFECT"         # Aligned + Compliant (1.0)
    ALIGNED = "ALIGNED"         # Aligned but non-compliant (0.85)
    WARNING = "WARNING"         # Aligned but should be compliant (0.7)
    VIOLATION = "VIOLATION"     # Not aligned (0.0)


@dataclass
class ReputationDelta:
    """
    R7 Framework: Explicit reputation changes from validation
    RFC-R6-TO-R7-EVOLUTION implementation
    """
    # Who
    subject_lct: str  # Whose reputation changed

    # What changed
    t3_delta: Dict[str, float] = field(default_factory=dict)  # Trust tensor changes
    v3_delta: Dict[str, float] = field(default_factory=dict)  # Value tensor changes

    # Why it changed
    reason: str = ""
    contributing_factors: List[str] = field(default_factory=list)

    # Who witnessed
    witnesses: List[str] = field(default_factory=list)

    # Magnitude
    net_trust_change: float = 0.0  # Sum of T3 deltas (-1.0 to +1.0)
    net_value_change: float = 0.0  # Sum of V3 deltas (-1.0 to +1.0)

    # Attribution
    action_id: str = ""
    rule_triggered: Optional[str] = None


@dataclass
class ComplianceRule:
    """
    Individual compliance rule with alignment/compliance distinction
    RFC-LAW-ALIGN-001 implementation
    """
    rule_id: str
    category: str  # "economic", "protocol", "training", "deployment"
    name: str
    severity: str  # "critical", "high", "medium", "low"
    description: str

    # NEW: Alignment (spirit of law)
    principle: str = ""  # WHY this law exists
    alignment_indicators: List[str] = field(default_factory=list)  # Observable behaviors

    # NEW: Compliance requirements (conditional)
    compliance_required: str = "always"  # "always", "conditional", "never"
    web4_level_requirements: Dict[str, str] = field(default_factory=dict)  # Level-specific requirements


class SAGEComplianceValidator:
    """
    Validates SAGE implementation against Web4 laws and protocols
    Society 4's Law Oracle contribution

    v2.0 Features:
    - Alignment (spirit) vs Compliance (letter) validation
    - R7 framework with explicit reputation tracking
    - Context-conditional compliance
    """

    def __init__(self, web4_level: int = 1):
        """
        Initialize validator

        Args:
            web4_level: Web4 abstraction level (0=physics, 1=virtual, 2=consensus)
        """
        self.web4_level = web4_level
        self.rules = self._define_rules()
        self.violations = []
        self.warnings = []
        self.reputation_deltas: List[ReputationDelta] = []

    def _define_rules(self) -> List[ComplianceRule]:
        """Define compliance rules with alignment/compliance distinction"""
        return [
            # Economic Laws
            ComplianceRule(
                rule_id="LAW-ECON-001",
                category="economic",
                name="Total ATP Budget Respected",
                severity="critical",
                description="Model must not exceed allocated ATP budget",
                principle="Systems must operate within finite resource constraints",
                alignment_indicators=[
                    "Resource consumption tracking exists",
                    "Hard limits enforced somewhere",
                    "Resource exhaustion handled gracefully"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "1000 ATP total budget in blockchain",
                    "1": "Virtual ATP tracking with configurable budget",
                    "0": "Physical power budget (watts) serves as ATP"
                }
            ),
            ComplianceRule(
                rule_id="LAW-ECON-003",
                category="economic",
                name="Daily Recharge Applied",
                severity="high",
                description="Model receives +20 ATP daily at 00:00 UTC",
                principle="Periodic resource regeneration prevents exhaustion",
                alignment_indicators=[
                    "Resources regenerate periodically",
                    "Regeneration prevents system starvation",
                    "Regeneration rate is predictable"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "+20 ATP at 00:00 UTC via blockchain BeginBlock",
                    "1": "Periodic recharge mechanism exists",
                    "0": "Continuous power supply provides effective recharge"
                }
            ),
            ComplianceRule(
                rule_id="PROC-ATP-DISCHARGE",
                category="economic",
                name="Energy Consumption Tracked",
                severity="high",
                description="All operations must record ATP discharge",
                principle="Resource consumption must be observable and accountable",
                alignment_indicators=[
                    "Operations have associated costs",
                    "Costs are tracked and recorded",
                    "Historical consumption is queryable"
                ],
                compliance_required="always"
            ),
            ComplianceRule(
                rule_id="ECON-CONSERVATION",
                category="economic",
                name="Energy Conservation",
                severity="medium",
                description="ATP + ADP should equal initial allocation",
                principle="Energy is conserved within system boundaries",
                alignment_indicators=[
                    "Total energy tracked",
                    "Energy transformations recorded",
                    "No energy created or destroyed"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "Exact conservation required",
                    "1": "Approximate conservation acceptable",
                    "0": "Physical power budget supersedes virtual tokens"
                }
            ),

            # Training Rules
            ComplianceRule(
                rule_id="TRAIN-ANTI-SHORTCUT",
                category="training",
                name="Anti-Shortcut Enforcement",
                severity="high",
                description="Training must penalize statistical shortcuts",
                principle="Learning should discover genuine patterns, not exploit dataset artifacts",
                alignment_indicators=[
                    "Model reasoning varies across tasks",
                    "Strategic thinking measurable (H-ratio variance)",
                    "Shortcuts detected and penalized"
                ],
                compliance_required="always"
            ),
            ComplianceRule(
                rule_id="TRAIN-REASONING-REWARD",
                category="training",
                name="Reasoning Over Accuracy",
                severity="high",
                description="Reward reasoning process, not just answers",
                principle="Understanding the process is more valuable than memorizing answers",
                alignment_indicators=[
                    "Partial credit awarded for reasoning",
                    "Process quality measured",
                    "Wrong answers with good reasoning rewarded"
                ],
                compliance_required="always"
            ),
            ComplianceRule(
                rule_id="TRAIN-ECONOMIC-PRESSURE",
                category="training",
                name="Economic Efficiency Pressure",
                severity="medium",
                description="Training should incentivize ATP efficiency",
                principle="Computational resources are finite and should be used efficiently",
                alignment_indicators=[
                    "Efficiency metrics tracked",
                    "Efficiency improves over time",
                    "Resource-reward ratio optimized"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "ATP efficiency tracked and rewarded",
                    "1": "Virtual resource efficiency tracked",
                    "0": "Power/performance ratio optimized"
                }
            ),

            # Protocol Rules
            ComplianceRule(
                rule_id="WEB4-IDENTITY",
                category="protocol",
                name="LCT Identity",
                severity="high",
                description="Components must have LCT identities",
                principle="All entities must have verifiable, unforgeable identity",
                alignment_indicators=[
                    "Entity can be uniquely identified",
                    "Identity cannot be forged",
                    "Identity persists across sessions"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "Full LCT with blockchain attestation",
                    "1": "Lightweight LCT with cryptographic binding",
                    "0": "Hardware serial number + MAC address"
                }
            ),
            ComplianceRule(
                rule_id="WEB4-WITNESS",
                category="protocol",
                name="Witness Attestation",
                severity="medium",
                description="Key events should be witnessed",
                principle="Trust emerges through observation and attestation",
                alignment_indicators=[
                    "Important events have observers",
                    "Observations are recorded",
                    "Multiple witnesses for critical events"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "Blockchain attestation required",
                    "1": "Cryptographic signatures from witnesses",
                    "0": "Audit logs sufficient"
                }
            ),
            ComplianceRule(
                rule_id="WEB4-TRUST",
                category="protocol",
                name="Trust Tensor Tracking",
                severity="low",
                description="T3/V3 tensors should track model trust",
                principle="Trust is the product, reputation must be explicit",
                alignment_indicators=[
                    "Trust changes tracked",
                    "Trust building visible",
                    "Reputation queryable"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "Full T3/V3 tensor tracking required",
                    "1": "Basic trust metrics tracked",
                    "0": "Performance history serves as implicit trust"
                }
            ),

            # Deployment Rules
            ComplianceRule(
                rule_id="DEPLOY-PERSISTENCE",
                category="deployment",
                name="State Persistence",
                severity="high",
                description="ATP state must persist across restarts",
                principle="State continuity prevents resource manipulation",
                alignment_indicators=[
                    "State survives restarts",
                    "No state manipulation possible",
                    "State recovery mechanism exists"
                ],
                compliance_required="always"
            ),
            ComplianceRule(
                rule_id="DEPLOY-MONITORING",
                category="deployment",
                name="Economic Monitoring",
                severity="medium",
                description="Real-time ATP/ADP monitoring available",
                principle="Observability enables trust and debugging",
                alignment_indicators=[
                    "Current state queryable",
                    "Historical data available",
                    "Monitoring accessible to operators"
                ],
                compliance_required="conditional",
                web4_level_requirements={
                    "2": "Real-time blockchain query API",
                    "1": "REST/RPC monitoring endpoint",
                    "0": "System metrics dashboard"
                }
            )
        ]

    def validate_training_run(self, training_log: Dict) -> Tuple[Dict, ReputationDelta]:
        """
        R7 Framework: Validate a training run and return Result + Reputation

        Args:
            training_log: Dict with training metrics and ATP transactions

        Returns:
            Tuple of (compliance_report, reputation_delta)
        """
        self.violations = []
        self.warnings = []
        self.reputation_deltas = []

        # Get subject LCT
        subject_lct = training_log.get("role_lct", "unknown")

        # Economic compliance checks
        self._check_atp_budget(training_log, subject_lct)
        self._check_daily_recharge(training_log, subject_lct)
        self._check_energy_tracking(training_log, subject_lct)
        self._check_conservation(training_log, subject_lct)

        # Training compliance checks
        self._check_anti_shortcut(training_log, subject_lct)
        self._check_reasoning_reward(training_log, subject_lct)
        self._check_economic_pressure(training_log, subject_lct)

        # Protocol compliance checks
        self._check_lct_identity(training_log, subject_lct)
        self._check_witnessing(training_log, subject_lct)
        self._check_trust_tensors(training_log, subject_lct)

        # Deployment compliance checks
        self._check_state_persistence(training_log, subject_lct)
        self._check_monitoring(training_log, subject_lct)

        # Generate R7 outputs
        report = self._generate_report()
        reputation = self._aggregate_reputation(
            subject_lct=subject_lct,
            report=report,
            witnesses=training_log.get("witnesses", []),
            action_id=training_log.get("action_id", f"validation_{datetime.now(timezone.utc).isoformat()}")
        )

        return report, reputation

    def _check_atp_budget(self, log: Dict, subject_lct: str):
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

    def _check_daily_recharge(self, log: Dict, subject_lct: str):
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

    def _check_energy_tracking(self, log: Dict, subject_lct: str):
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

    def _check_conservation(self, log: Dict, subject_lct: str):
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

    def _check_anti_shortcut(self, log: Dict, subject_lct: str):
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

    def _check_reasoning_reward(self, log: Dict, subject_lct: str):
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

    def _check_economic_pressure(self, log: Dict, subject_lct: str):
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

    def _check_lct_identity(self, log: Dict, subject_lct: str):
        """WEB4-IDENTITY: LCT Identity"""
        role_lct = log.get("role_lct", "")

        if not role_lct or not role_lct.startswith("lct:web4:"):
            self.violations.append({
                "rule": "WEB4-IDENTITY",
                "severity": "high",
                "message": "No valid LCT identity",
                "details": f"role_lct: {role_lct}"
            })

    def _check_witnessing(self, log: Dict, subject_lct: str):
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

    def _check_trust_tensors(self, log: Dict, subject_lct: str):
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

    def _check_state_persistence(self, log: Dict, subject_lct: str):
        """DEPLOY-PERSISTENCE: State Persistence"""
        state_file = log.get("state_file", "")

        if not state_file:
            self.violations.append({
                "rule": "DEPLOY-PERSISTENCE",
                "severity": "high",
                "message": "No state persistence configured",
                "details": "ATP state will be lost on restart"
            })

    def _check_monitoring(self, log: Dict, subject_lct: str):
        """DEPLOY-MONITORING: Economic Monitoring"""
        monitoring = log.get("monitoring_enabled", False)

        if not monitoring:
            self.warnings.append({
                "rule": "DEPLOY-MONITORING",
                "severity": "medium",
                "message": "No economic monitoring",
                "details": "Real-time ATP/ADP tracking not available"
            })

    def _aggregate_reputation(
        self,
        subject_lct: str,
        report: Dict,
        witnesses: List[str],
        action_id: str
    ) -> ReputationDelta:
        """
        R7 Framework: Aggregate reputation changes from validation

        Args:
            subject_lct: LCT being validated
            report: Compliance report
            witnesses: List of witness LCTs
            action_id: Unique action identifier

        Returns:
            ReputationDelta with explicit trust/value changes
        """
        t3_delta = {}
        v3_delta = {}
        contributing_factors = []

        # Calculate reputation changes based on compliance
        compliance_score = report["compliance_score"]
        violations = report["violations"]

        # Technical competence (T3 dimension)
        if violations["critical"]:
            t3_delta["technical_competence"] = -0.10  # Critical violations hurt trust
            contributing_factors.append(f"{len(violations['critical'])} critical violations")
        elif violations["high"]:
            t3_delta["technical_competence"] = -0.03
            contributing_factors.append(f"{len(violations['high'])} high-severity violations")
        elif compliance_score >= 0.95:
            t3_delta["technical_competence"] = +0.05  # Excellent compliance builds trust
            contributing_factors.append("Excellent compliance (95%+)")
        elif compliance_score >= 0.8:
            t3_delta["technical_competence"] = +0.02  # Good compliance
            contributing_factors.append("Good compliance (80%+)")

        # Social reliability (T3 dimension) - based on warnings
        warning_count = len(report["warnings"])
        if warning_count == 0 and compliance_score >= 0.9:
            t3_delta["social_reliability"] = +0.03  # No warnings = reliable
            contributing_factors.append("Zero warnings, highly reliable")
        elif warning_count > 5:
            t3_delta["social_reliability"] = -0.02  # Many warnings = less reliable
            contributing_factors.append(f"{warning_count} warnings indicate reliability concerns")

        # Resource stewardship (V3 dimension) - economic behavior
        has_economic_violations = any(
            v["rule"].startswith("LAW-ECON") or v["rule"].startswith("PROC-ATP")
            for v in violations["critical"] + violations["high"] + violations["medium"]
        )
        if not has_economic_violations and compliance_score >= 0.8:
            v3_delta["resource_stewardship"] = +0.04  # Good economic behavior
            contributing_factors.append("Excellent resource management")
        elif has_economic_violations:
            v3_delta["resource_stewardship"] = -0.05  # Poor economic behavior
            contributing_factors.append("Economic law violations detected")

        # Contribution history (V3 dimension) - passing validation is contributing
        if compliance_score >= 0.8:
            v3_delta["contribution_history"] = +0.02
            contributing_factors.append("Successful validation contributes to ecosystem")

        # Calculate net changes
        net_trust_change = sum(t3_delta.values())
        net_value_change = sum(v3_delta.values())

        # Generate human-readable reason
        if compliance_score >= 0.95:
            reason = "Excellent Web4 compliance with all laws honored"
        elif compliance_score >= 0.8:
            reason = "Good Web4 compliance meets production standards"
        elif violations["critical"]:
            reason = f"Critical violations of {len(violations['critical'])} Web4 laws"
        else:
            reason = f"Partial compliance ({compliance_score:.1%}) below production threshold"

        # Determine which rules triggered reputation changes
        triggered_rules = []
        if violations["critical"]:
            triggered_rules.extend([v["rule"] for v in violations["critical"]])
        if violations["high"]:
            triggered_rules.extend([v["rule"] for v in violations["high"][:3]])  # Top 3

        return ReputationDelta(
            subject_lct=subject_lct,
            t3_delta=t3_delta,
            v3_delta=v3_delta,
            reason=reason,
            contributing_factors=contributing_factors,
            witnesses=witnesses if witnesses else ["lct:web4:society:4:law-oracle"],
            net_trust_change=net_trust_change,
            net_value_change=net_value_change,
            action_id=action_id,
            rule_triggered=triggered_rules[0] if triggered_rules else None
        )

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
    print("=" * 80)
    print("SAGE Compliance Validator v2.0 - Society 4 Law Oracle")
    print("RFC-LAW-ALIGN-001 + RFC-R6-TO-R7-EVOLUTION Implementation")
    print("=" * 80)

    # Web4 Level 1 (Virtual) - SAGE running with economic wrapper
    validator = SAGEComplianceValidator(web4_level=1)

    # Simulate training log
    training_log = {
        "role_lct": "lct:web4:society:federation:sage_model",
        "action_id": "sage_training_run_001",
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
        "witnesses": ["lct:web4:society:4:law-oracle", "lct:web4:genesis:sage"],
        "t3_tensor": {},
        "v3_tensor": {}
    }

    print("\nValidating SAGE training run (R7 Framework)...")
    print("-" * 80)

    # R7 Framework: Returns (Result, Reputation)
    report, reputation = validator.validate_training_run(training_log)

    # Display Result (compliance report)
    print("\n📋 RESULT (Compliance Report):")
    print("-" * 80)
    print(f"Compliance Score: {report['compliance_score']:.1%}")
    print(f"Status: {report['summary']}")
    print(f"Passed Rules: {report['passed_rules']}/{report['total_rules']}")

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

    # Display Reputation (R7 framework explicit output)
    print("\n⭐ REPUTATION (R7 Framework Explicit Output):")
    print("-" * 80)
    print(f"Subject: {reputation.subject_lct}")
    print(f"Action: {reputation.action_id}")
    print(f"Witnesses: {len(reputation.witnesses)}")

    print(f"\nTrust Changes (T3 Tensor):")
    for dimension, delta in reputation.t3_delta.items():
        print(f"  {dimension}: {delta:+.3f}")
    print(f"  NET TRUST CHANGE: {reputation.net_trust_change:+.3f}")

    print(f"\nValue Changes (V3 Tensor):")
    for dimension, delta in reputation.v3_delta.items():
        print(f"  {dimension}: {delta:+.3f}")
    print(f"  NET VALUE CHANGE: {reputation.net_value_change:+.3f}")

    print(f"\nReason: {reputation.reason}")
    print(f"\nContributing Factors:")
    for factor in reputation.contributing_factors:
        print(f"  • {factor}")

    print("\n" + "=" * 80)
    print("✅ R7 Validation Complete: Result + Reputation returned")
    print("Trust-building is now explicit and traceable!")
    print("=" * 80)
