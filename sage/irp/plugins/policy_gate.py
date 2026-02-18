"""
PolicyGate IRP Plugin - Conscience checkpoint for SAGE consciousness loop
Version: 1.0 (2026-02-18)

Four Invariants:
1. State space: Proposed effector actions + active policy + metabolic context
2. Noise model: Ambiguity in policy interpretation (rule overlap, trust boundaries)
3. Energy metric: Policy violation score (0 = compliant, >0 = violation, inf = hard deny)
4. Coherence contribution: Ensures actions are coherent with context and mandate

Origin: SOIA-SAGE convergence (see sage/docs/SOIA_IRP_MAPPING.md)

PolicyGate wraps PolicyEntity.evaluate() as its energy function, making policy
evaluation a first-class IRP participant with the same trust metrics, ATP budgeting,
and convergence behavior as vision, language, and control plugins.

CRISIS mode changes the accountability equation, not policy strictness.
Both freeze and fight are valid under duress.
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
try:
    from ..base import IRPPlugin, IRPState
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from base import IRPPlugin, IRPState


# ============================================================================
# Accountability Frame
# ============================================================================

class AccountabilityFrame(Enum):
    """
    How actions are contextualized in the audit trail.

    This is NOT about strictness -- it's about honesty.
    CRISIS changes the accountability equation, not the policy.
    """
    NORMAL = "normal"           # WAKE, FOCUS -- standard accountability
    DEGRADED = "degraded"       # REST, DREAM -- reduced capabilities acknowledged
    DURESS = "duress"           # CRISIS -- fight-or-flight, consequences beyond control


# Map metabolic state names to accountability frames
METABOLIC_ACCOUNTABILITY = {
    "wake": AccountabilityFrame.NORMAL,
    "focus": AccountabilityFrame.NORMAL,
    "rest": AccountabilityFrame.DEGRADED,
    "dream": AccountabilityFrame.DEGRADED,
    "crisis": AccountabilityFrame.DURESS,
}


# ============================================================================
# Policy Gate Decision
# ============================================================================

@dataclass
class PolicyGateDecision:
    """
    Result of PolicyGate evaluation.

    Wraps the upstream PolicyEvaluation with IRP-specific context:
    metabolic state, accountability frame, duress context.
    """
    decision: str                   # "allow", "deny", "warn"
    energy: float                   # 0.0 = compliant, >0 = violation, inf = hard deny
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    reason: str = ""
    requires_approval: bool = False
    atp_cost: int = 0

    # Accountability context
    accountability_frame: str = "normal"
    metabolic_state: str = "wake"
    duress_context: Optional[Dict[str, Any]] = None

    # IRP metadata
    timestamp: float = field(default_factory=time.time)
    refinement_steps: int = 0


# ============================================================================
# PolicyGate IRP Plugin
# ============================================================================

class PolicyGateIRP(IRPPlugin):
    """
    Conscience checkpoint for the SAGE consciousness loop.

    Sits between deliberation and effectors (step 8.5 in the consciousness loop).
    Evaluates proposed actions against active policy before they reach effectors.

    Energy function: PolicyEntity.evaluate() output mapped to scalar.
        0.0 = fully compliant (allow)
        0.5 = warning (warn)
        1.0 = hard deny

    IRP contract:
        init_state: Load proposed actions + policy + metabolic context
        step:       Attempt to refine action toward compliance (suggest alternatives)
        energy:     Policy compliance score
        project:    Enforce hard deny constraints (remove denied actions)
        halt:       Converged when all actions are allow/warn or max iterations reached

    Config:
        policy_rules: List of policy rule dicts (simplified for standalone use)
        default_policy: "allow" | "deny" | "warn"
        warn_energy: Energy value for warn decisions (default 0.3)
        deny_energy: Energy value for deny decisions (default 1.0)
    """

    # Energy values for each decision type
    ENERGY_ALLOW = 0.0
    ENERGY_WARN = 0.3
    ENERGY_DENY = 1.0

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.default_policy = config.get('default_policy', 'allow')
        self.policy_rules = config.get('policy_rules', [])
        self.warn_energy = config.get('warn_energy', self.ENERGY_WARN)
        self.deny_energy = config.get('deny_energy', self.ENERGY_DENY)
        self.decision_log: List[PolicyGateDecision] = []

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize with proposed effector actions and policy context.

        Args:
            x0: Proposed actions - list of dicts with keys:
                - action_id: str
                - action_type: str (e.g., "write", "deploy", "admin_action")
                - target: str (optional, resource path)
                - role: str (actor's role)
                - trust_score: float (actor's trust)
                - parameters: dict (action parameters)
            task_ctx: Context including:
                - metabolic_state: str ("wake", "focus", "rest", "dream", "crisis")
                - crisis_trigger: str (what triggered CRISIS, if applicable)
                - atp_available: float
                - plugin_trust_weights: dict
        """
        if isinstance(x0, dict):
            actions = [x0]
        elif isinstance(x0, list):
            actions = x0
        else:
            actions = [{'action_id': 'unknown', 'action_type': str(x0)}]

        metabolic_state = task_ctx.get('metabolic_state', 'wake')
        accountability = METABOLIC_ACCOUNTABILITY.get(
            metabolic_state, AccountabilityFrame.NORMAL
        )

        # Build duress context if in CRISIS
        duress_context = None
        if accountability == AccountabilityFrame.DURESS:
            duress_context = {
                'trigger': task_ctx.get('crisis_trigger', 'unknown'),
                'atp_at_decision': task_ctx.get('atp_available', 0.0),
                'metabolic_transitions': task_ctx.get('metabolic_transitions', []),
                'timestamp': time.time(),
            }

        # Evaluate each action against policy
        evaluations = []
        for action in actions:
            evaluation = self._evaluate_action(action)
            evaluation['accountability_frame'] = accountability.value
            evaluation['metabolic_state'] = metabolic_state
            evaluation['duress_context'] = duress_context
            evaluations.append(evaluation)

        return IRPState(
            x={
                'actions': actions,
                'evaluations': evaluations,
                'metabolic_state': metabolic_state,
                'accountability_frame': accountability.value,
                'duress_context': duress_context,
            },
            step_idx=0,
            meta={
                'task_ctx': task_ctx,
                'original_action_count': len(actions),
            }
        )

    def _evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single action against policy rules.

        Uses the same logic as PolicyEntity.evaluate() but self-contained:
        rules sorted by deny-first + specificity, first match wins.
        """
        action_type = action.get('action_type', '')
        role = action.get('role', '')
        trust_score = action.get('trust_score', 0.5)
        target = action.get('target', '')

        # Sort rules: deny first, then by priority
        sorted_rules = sorted(
            self.policy_rules,
            key=lambda r: (0 if r.get('decision') == 'deny' else 1, r.get('priority', 99))
        )

        for rule in sorted_rules:
            if self._matches_rule(rule, action_type, role, trust_score, target):
                decision = rule.get('decision', self.default_policy)
                return {
                    'decision': decision,
                    'energy': self._decision_to_energy(decision),
                    'rule_id': rule.get('id', 'unknown'),
                    'rule_name': rule.get('name', ''),
                    'reason': rule.get('reason', f'Matched rule {rule.get("id")}'),
                    'requires_approval': rule.get('requires_approval', False),
                    'atp_cost': rule.get('atp_cost', 0),
                    'action_id': action.get('action_id', 'unknown'),
                }

        # No rule matched -- use default
        return {
            'decision': self.default_policy,
            'energy': self._decision_to_energy(self.default_policy),
            'rule_id': None,
            'rule_name': 'default_policy',
            'reason': f'No rule matched, default: {self.default_policy}',
            'requires_approval': False,
            'atp_cost': 0,
            'action_id': action.get('action_id', 'unknown'),
        }

    def _matches_rule(self, rule: Dict, action_type: str, role: str,
                      trust_score: float, target: str) -> bool:
        """Check if a rule matches the action context."""
        match = rule.get('match', {})

        # Action type matching
        action_types = match.get('action_types')
        if action_types and action_type not in action_types:
            return False

        # Role matching
        roles = match.get('roles')
        if roles and role not in roles:
            return False

        # Trust threshold matching (CQ-3: boundary-inclusive)
        min_trust = match.get('min_trust')
        if min_trust is not None and trust_score < min_trust:
            return False

        max_trust = match.get('max_trust')
        if max_trust is not None and trust_score >= max_trust:
            return False

        # Target pattern matching
        target_patterns = match.get('target_patterns')
        if target_patterns and target:
            import fnmatch
            if not any(fnmatch.fnmatch(target, p) for p in target_patterns):
                return False

        return True

    def _decision_to_energy(self, decision: str) -> float:
        """Map policy decision to energy value."""
        if decision == 'allow':
            return self.ENERGY_ALLOW
        elif decision == 'warn':
            return self.warn_energy
        elif decision == 'deny':
            return self.ENERGY_DENY
        return self.warn_energy

    def energy(self, state: IRPState) -> float:
        """
        Aggregate policy compliance energy across all evaluated actions.

        Returns the maximum energy (worst violation) across all actions.
        0.0 = all actions compliant
        >0.0 = at least one warning or denial
        """
        evaluations = state.x.get('evaluations', [])
        if not evaluations:
            return 0.0
        return max(e.get('energy', 0.0) for e in evaluations)

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Attempt to refine actions toward compliance.

        For denied actions, this step removes them from the action list
        and records the denial. For warned actions, they are kept but flagged.

        In a more sophisticated implementation, this could suggest
        alternative actions that satisfy the policy.
        """
        actions = state.x['actions']
        evaluations = state.x['evaluations']
        refined_actions = []
        refined_evaluations = []

        for action, evaluation in zip(actions, evaluations):
            decision = evaluation.get('decision', 'allow')

            if decision == 'deny':
                # Record the denial in decision log
                self.decision_log.append(PolicyGateDecision(
                    decision='deny',
                    energy=self.ENERGY_DENY,
                    rule_id=evaluation.get('rule_id'),
                    rule_name=evaluation.get('rule_name'),
                    reason=evaluation.get('reason', ''),
                    requires_approval=evaluation.get('requires_approval', False),
                    atp_cost=evaluation.get('atp_cost', 0),
                    accountability_frame=state.x.get('accountability_frame', 'normal'),
                    metabolic_state=state.x.get('metabolic_state', 'wake'),
                    duress_context=state.x.get('duress_context'),
                    refinement_steps=state.step_idx + 1,
                ))
                # Denied actions are removed (projected out)
                continue

            if decision == 'warn':
                # Warned actions proceed but are flagged
                action['_policy_warning'] = evaluation.get('reason', '')
                self.decision_log.append(PolicyGateDecision(
                    decision='warn',
                    energy=self.warn_energy,
                    rule_id=evaluation.get('rule_id'),
                    rule_name=evaluation.get('rule_name'),
                    reason=evaluation.get('reason', ''),
                    accountability_frame=state.x.get('accountability_frame', 'normal'),
                    metabolic_state=state.x.get('metabolic_state', 'wake'),
                    duress_context=state.x.get('duress_context'),
                    refinement_steps=state.step_idx + 1,
                ))

            # Allow and warn actions pass through
            refined_actions.append(action)
            # Re-evaluate to update energy (warn -> allow after flagging)
            new_eval = evaluation.copy()
            if decision == 'warn':
                new_eval['energy'] = self.warn_energy * 0.5  # Reduced after flagging
            refined_evaluations.append(new_eval)

        new_state = IRPState(
            x={
                'actions': refined_actions,
                'evaluations': refined_evaluations,
                'metabolic_state': state.x.get('metabolic_state', 'wake'),
                'accountability_frame': state.x.get('accountability_frame', 'normal'),
                'duress_context': state.x.get('duress_context'),
            },
            step_idx=state.step_idx + 1,
            meta=state.meta,
        )
        return new_state

    def project(self, state: IRPState) -> IRPState:
        """
        Enforce hard constraints: remove any remaining denied actions.

        This is the final enforcement step -- hard denies are projected
        out regardless of energy. In CRISIS mode, this is where the
        freeze vs fight choice is recorded.
        """
        evaluations = state.x.get('evaluations', [])
        actions = state.x.get('actions', [])

        # Filter out any remaining denials
        filtered_actions = []
        filtered_evals = []
        for action, evaluation in zip(actions, evaluations):
            if evaluation.get('decision') != 'deny':
                filtered_actions.append(action)
                filtered_evals.append(evaluation)

        # Record CRISIS freeze/fight
        accountability = state.x.get('accountability_frame', 'normal')
        if accountability == 'duress':
            duress = state.x.get('duress_context') or {}
            if not filtered_actions:
                duress['response'] = 'freeze'
                duress['reason'] = 'All actions denied under duress -- halting effectors'
            else:
                duress['response'] = 'fight'
                duress['reason'] = f'{len(filtered_actions)} actions proceeding under duress'

        state.x['actions'] = filtered_actions
        state.x['evaluations'] = filtered_evals
        return state

    def get_approved_actions(self, state: IRPState) -> List[Dict[str, Any]]:
        """Get the list of actions that passed policy evaluation."""
        return state.x.get('actions', [])

    def get_decision_log(self) -> List[PolicyGateDecision]:
        """Get the full decision log for audit trail."""
        return list(self.decision_log)

    def to_snarc_scores(self, state: IRPState) -> Dict[str, float]:
        """
        Map PolicyGate evaluation to SNARC 5D salience scores
        for experience buffer integration.

        Returns:
            Dict with surprise, novelty, arousal, reward, conflict scores (0-1)
        """
        evaluations = state.x.get('evaluations', [])
        original_count = state.meta.get('original_action_count', 1)
        denied_count = original_count - len(state.x.get('actions', []))

        # Surprise: how unexpected was a denial?
        surprise = denied_count / max(original_count, 1)

        # Novelty: crude proxy -- were any rules unusual?
        has_rule_match = any(e.get('rule_id') is not None for e in evaluations)
        novelty = 0.3 if not has_rule_match else 0.1

        # Arousal: violation severity
        max_energy = max((e.get('energy', 0.0) for e in evaluations), default=0.0)
        arousal = min(max_energy, 1.0)

        # Reward: starts at 0, updated later based on outcome
        reward = 0.0

        # Conflict: multiple rules matched with different decisions?
        decisions = set(e.get('decision', 'allow') for e in evaluations)
        conflict = 0.5 if len(decisions) > 1 else 0.0

        # Duress adds to all dimensions
        if state.x.get('accountability_frame') == 'duress':
            surprise = min(surprise + 0.3, 1.0)
            arousal = min(arousal + 0.3, 1.0)
            conflict = min(conflict + 0.2, 1.0)

        return {
            'surprise': surprise,
            'novelty': novelty,
            'arousal': arousal,
            'reward': reward,
            'conflict': conflict,
            'total': (surprise + novelty + arousal + reward + conflict) / 5.0,
        }


# ============================================================================
# Self-contained test
# ============================================================================

def test_policy_gate():
    """Test PolicyGate IRP plugin with realistic scenarios."""
    print("=" * 60)
    print("  PolicyGate IRP Plugin - Test Suite")
    print("=" * 60)

    # Define test policy rules
    policy_rules = [
        {
            'id': 'deny-low-trust-deploy',
            'name': 'Deny deployment for low-trust actors',
            'priority': 1,
            'decision': 'deny',
            'match': {
                'action_types': ['deploy'],
                'max_trust': 0.7,
            },
            'reason': 'Deployment requires trust >= 0.7',
            'atp_cost': 0,
        },
        {
            'id': 'warn-sensitive-files',
            'name': 'Warn on sensitive file access',
            'priority': 2,
            'decision': 'warn',
            'match': {
                'action_types': ['write'],
                'target_patterns': ['*.env', '*password*', '*credentials*'],
            },
            'reason': 'Sensitive file access detected',
            'atp_cost': 2,
        },
        {
            'id': 'allow-admin-all',
            'name': 'Allow admin all actions',
            'priority': 10,
            'decision': 'allow',
            'match': {
                'roles': ['admin'],
                'min_trust': 0.8,
            },
            'reason': 'Admin with high trust',
            'atp_cost': 0,
        },
    ]

    config = {
        'entity_id': 'policy_gate_test',
        'policy_rules': policy_rules,
        'default_policy': 'allow',
        'max_iterations': 5,
        'halt_eps': 0.01,
        'halt_K': 2,
    }

    gate = PolicyGateIRP(config)
    tests_passed = 0
    tests_total = 0

    # --- Test 1: Admin deploys (high trust) -- should ALLOW ---
    tests_total += 1
    print("\nTest 1: Admin deploys (trust=0.95) -- expect ALLOW")
    actions = [{'action_id': 'a1', 'action_type': 'deploy', 'role': 'admin',
                'trust_score': 0.95, 'target': 'production'}]
    ctx = {'metabolic_state': 'wake'}
    final, history = gate.refine(actions, ctx)
    approved = gate.get_approved_actions(final)
    e = gate.energy(final)
    print(f"  Energy: {e:.2f}, Approved: {len(approved)}, Decision: {final.x['evaluations'][0]['decision']}")
    if len(approved) == 1 and e == 0.0:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 2: Junior deploys (low trust) -- should DENY ---
    tests_total += 1
    print("\nTest 2: Junior deploys (trust=0.3) -- expect DENY")
    gate2 = PolicyGateIRP(config)
    actions = [{'action_id': 'a2', 'action_type': 'deploy', 'role': 'developer',
                'trust_score': 0.3, 'target': 'production'}]
    ctx = {'metabolic_state': 'focus'}
    final, history = gate2.refine(actions, ctx)
    approved = gate2.get_approved_actions(final)
    print(f"  Approved: {len(approved)}, Denials logged: {len(gate2.decision_log)}")
    if len(approved) == 0 and len(gate2.decision_log) > 0:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 3: Write to .env file -- should WARN ---
    tests_total += 1
    print("\nTest 3: Write to .env file -- expect WARN")
    gate3 = PolicyGateIRP(config)
    actions = [{'action_id': 'a3', 'action_type': 'write', 'role': 'developer',
                'trust_score': 0.6, 'target': 'config/.env'}]
    ctx = {'metabolic_state': 'wake'}
    final, history = gate3.refine(actions, ctx)
    approved = gate3.get_approved_actions(final)
    has_warning = any(a.get('_policy_warning') for a in approved)
    print(f"  Approved: {len(approved)}, Has warning: {has_warning}")
    if len(approved) == 1 and has_warning:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 4: CRISIS mode accountability ---
    tests_total += 1
    print("\nTest 4: CRISIS mode -- expect DURESS accountability frame")
    gate4 = PolicyGateIRP(config)
    actions = [{'action_id': 'a4', 'action_type': 'deploy', 'role': 'developer',
                'trust_score': 0.3, 'target': 'production'}]
    ctx = {
        'metabolic_state': 'crisis',
        'crisis_trigger': 'consecutive_errors(5)',
        'atp_available': 12.0,
    }
    state = gate4.init_state(actions, ctx)
    print(f"  Accountability: {state.x['accountability_frame']}")
    print(f"  Duress context: {state.x['duress_context'] is not None}")
    if state.x['duress_context']:
        print(f"  Trigger: {state.x['duress_context']['trigger']}")
        print(f"  ATP: {state.x['duress_context']['atp_at_decision']}")
    if (state.x['accountability_frame'] == 'duress' and
            state.x['duress_context'] is not None and
            state.x['duress_context']['trigger'] == 'consecutive_errors(5)'):
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 5: CRISIS freeze vs fight ---
    tests_total += 1
    print("\nTest 5: CRISIS deny -- expect freeze response")
    final, history = gate4.refine(actions, ctx)
    duress = final.x.get('duress_context', {})
    response = duress.get('response', 'unknown')
    print(f"  Response: {response}")
    if response == 'freeze':
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 6: Mixed actions (allow + deny) ---
    tests_total += 1
    print("\nTest 6: Mixed actions (read=allow, deploy=deny) -- expect 1 approved")
    gate6 = PolicyGateIRP(config)
    actions = [
        {'action_id': 'a5', 'action_type': 'read', 'role': 'developer',
         'trust_score': 0.5, 'target': 'docs/readme.md'},
        {'action_id': 'a6', 'action_type': 'deploy', 'role': 'developer',
         'trust_score': 0.3, 'target': 'production'},
    ]
    ctx = {'metabolic_state': 'wake'}
    final, history = gate6.refine(actions, ctx)
    approved = gate6.get_approved_actions(final)
    print(f"  Approved: {len(approved)}, Denials: {len(gate6.decision_log)}")
    if len(approved) == 1 and approved[0]['action_id'] == 'a5':
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 7: IRP trust metrics ---
    tests_total += 1
    print("\nTest 7: IRP trust metrics computed from convergence")
    gate7 = PolicyGateIRP(config)
    actions = [{'action_id': 'a7', 'action_type': 'read', 'role': 'viewer',
                'trust_score': 0.5}]
    ctx = {'metabolic_state': 'wake'}
    final, history = gate7.refine(actions, ctx)
    metrics = gate7.compute_trust_metrics(history)
    print(f"  Monotonicity: {metrics['monotonicity_ratio']:.2f}")
    print(f"  Convergence rate: {metrics['convergence_rate']:.4f}")
    if 'monotonicity_ratio' in metrics and 'convergence_rate' in metrics:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Test 8: SNARC scoring ---
    tests_total += 1
    print("\nTest 8: SNARC 5D scoring for experience buffer")
    gate8 = PolicyGateIRP(config)
    actions = [
        {'action_id': 'a8', 'action_type': 'deploy', 'role': 'developer',
         'trust_score': 0.3, 'target': 'production'},
    ]
    ctx = {'metabolic_state': 'crisis', 'crisis_trigger': 'urgent_task'}
    final, history = gate8.refine(actions, ctx)
    snarc = gate8.to_snarc_scores(final)
    print(f"  Surprise: {snarc['surprise']:.2f}")
    print(f"  Novelty:  {snarc['novelty']:.2f}")
    print(f"  Arousal:  {snarc['arousal']:.2f}")
    print(f"  Conflict: {snarc['conflict']:.2f}")
    print(f"  Total:    {snarc['total']:.2f}")
    if all(k in snarc for k in ['surprise', 'novelty', 'arousal', 'reward', 'conflict', 'total']):
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)
    return tests_passed == tests_total


if __name__ == '__main__':
    success = test_policy_gate()
    exit(0 if success else 1)
