"""
Federation Challenge System for SAGE Consciousness

Adapted from Web4 Session #84 Challenge Evasion Defense for consciousness
federation. Platforms must respond to execution quality challenges within
timeout or face progressive reputation penalties.

Problem:
- Platforms could delegate tasks but provide low-quality results
- Platforms could go offline when challenged about quality
- No temporal accountability for maintaining reputation

Solution:
- Quality challenges with 24h timeout
- Progressive penalties for evasion (5% → 50% reputation decay)
- Strike system (3 strikes → severe penalty)
- Re-challenge cooldown to prevent spam

Integration with Federation:
- ExecutionProof quality can be challenged
- Witnesses can issue challenges
- Platform must re-execute or provide evidence
- Reputation updated based on challenge outcome

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - ATP-Security Integration
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict

# Import federation types
from sage.federation.federation_types import FederationIdentity, ExecutionProof


# ============================================================================
# Challenge Status and Penalty Types
# ============================================================================

class ChallengeStatus(Enum):
    """Status of quality challenge"""
    PENDING = "pending"      # Challenge issued, awaiting response
    RESPONDED = "responded"  # Platform responded with evidence
    EVADED = "evaded"        # Platform evaded (timeout expired)
    EXPIRED = "expired"      # Challenge expired after evasion


class EvasionPenaltyLevel(Enum):
    """Severity of evasion penalty"""
    NONE = 0           # No penalty (new platform or legitimate)
    WARNING = 1        # Warning level (5% decay)
    MODERATE = 2       # Moderate penalty (15% decay)
    SEVERE = 3         # Severe penalty (30% decay)
    PERMANENT = 4      # Permanent reduction (50% decay, 3+ strikes)


# ============================================================================
# Challenge Data Structures
# ============================================================================

@dataclass
class QualityChallenge:
    """
    Challenge to platform's execution quality claim

    When a platform submits an ExecutionProof with claimed quality,
    witnesses or other platforms can challenge this claim. The
    platform must provide evidence or re-execute within timeout.
    """
    challenge_id: str
    platform_lct_id: str          # Platform being challenged
    challenger_lct_id: str        # Platform issuing challenge
    challenged_proof_id: str      # ExecutionProof being challenged
    claimed_quality: float        # Claimed quality score (0-1)
    timeout_period: float = 86400.0  # Default: 24 hours
    cooldown_period: float = 604800.0  # Default: 7 days before re-challenge

    issue_timestamp: float = field(default_factory=time.time)
    response_timestamp: Optional[float] = None
    timeout_timestamp: float = field(init=False)
    status: ChallengeStatus = ChallengeStatus.PENDING

    # Response details
    response_evidence: Optional[Dict[str, Any]] = None
    verified_quality: Optional[float] = None  # Quality after verification
    response_verified: bool = False

    def __post_init__(self):
        self.timeout_timestamp = self.issue_timestamp + self.timeout_period

    def has_timed_out(self, current_time: Optional[float] = None) -> bool:
        """Check if challenge has timed out"""
        current_time = current_time or time.time()
        return (current_time >= self.timeout_timestamp and
                self.status == ChallengeStatus.PENDING)

    def respond(self,
                evidence: Dict[str, Any],
                current_time: Optional[float] = None) -> bool:
        """
        Platform responds to challenge with evidence

        Args:
            evidence: Evidence supporting quality claim (re-execution results, etc.)
            current_time: Current timestamp

        Returns:
            True if response accepted (within timeout), False if too late
        """
        current_time = current_time or time.time()

        if current_time > self.timeout_timestamp:
            # Too late, already timed out
            return False

        self.response_timestamp = current_time
        self.response_evidence = evidence
        self.status = ChallengeStatus.RESPONDED
        return True

    def verify_response(self, verified_quality: float, is_valid: bool):
        """
        Verify platform's response

        Args:
            verified_quality: Measured quality from re-execution/verification
            is_valid: Whether response was valid
        """
        if self.status != ChallengeStatus.RESPONDED:
            return

        self.verified_quality = verified_quality
        self.response_verified = is_valid

    def mark_evaded(self):
        """Mark challenge as evaded (timeout expired)"""
        self.status = ChallengeStatus.EVADED


@dataclass
class EvasionRecord:
    """
    Per-platform evasion tracking

    Tracks challenge evasion history to determine progressive penalties.
    Based on Web4's progressive penalty system.
    """
    platform_lct_id: str
    total_challenges: int = 0
    responded_challenges: int = 0
    evaded_challenges: int = 0
    strike_count: int = 0

    # Temporal tracking
    first_challenge: Optional[float] = None
    last_challenge: Optional[float] = None
    last_evasion: Optional[float] = None

    # Quality tracking
    average_verified_quality: float = 0.5  # Start neutral
    total_verifications: int = 0

    def add_challenge(self, challenge: QualityChallenge):
        """Record challenge outcome"""
        self.total_challenges += 1

        if self.first_challenge is None:
            self.first_challenge = challenge.issue_timestamp
        self.last_challenge = challenge.issue_timestamp

        if challenge.status == ChallengeStatus.RESPONDED:
            self.responded_challenges += 1

            # Update quality tracking if verified
            if challenge.verified_quality is not None:
                self.total_verifications += 1
                # Exponential moving average
                alpha = 0.3
                self.average_verified_quality = (
                    alpha * challenge.verified_quality +
                    (1 - alpha) * self.average_verified_quality
                )

        elif challenge.status == ChallengeStatus.EVADED:
            self.evaded_challenges += 1
            self.last_evasion = challenge.timeout_timestamp
            self.strike_count += 1

    def get_response_rate(self) -> float:
        """Calculate response rate (responded / total)"""
        if self.total_challenges == 0:
            return 1.0  # Benefit of doubt for new platforms
        return self.responded_challenges / self.total_challenges

    def get_evasion_rate(self) -> float:
        """Calculate evasion rate (evaded / total)"""
        if self.total_challenges == 0:
            return 0.0
        return self.evaded_challenges / self.total_challenges

    def get_penalty_level(self) -> EvasionPenaltyLevel:
        """Determine penalty level based on strike count"""
        if self.strike_count == 0:
            return EvasionPenaltyLevel.NONE
        elif self.strike_count == 1:
            return EvasionPenaltyLevel.WARNING
        elif self.strike_count == 2:
            return EvasionPenaltyLevel.MODERATE
        elif self.strike_count == 3:
            return EvasionPenaltyLevel.SEVERE
        else:  # 4+
            return EvasionPenaltyLevel.PERMANENT


# ============================================================================
# Federation Challenge System
# ============================================================================

class FederationChallengeSystem:
    """
    Main system for federation quality challenge defense

    Manages quality challenges, tracks evasion, applies progressive
    reputation penalties for SAGE consciousness platforms.

    Based on Web4 Session #84 challenge_evasion_defense.py adapted
    for consciousness federation requirements.
    """

    def __init__(self,
                 default_timeout: float = 86400.0,     # 24 hours
                 re_challenge_cooldown: float = 604800.0,  # 7 days
                 decay_rates: Optional[Dict[EvasionPenaltyLevel, float]] = None):
        """
        Initialize federation challenge system

        Args:
            default_timeout: Default challenge timeout (seconds)
            re_challenge_cooldown: Cooldown before re-challenge (seconds)
            decay_rates: Reputation decay per penalty level
        """
        self.default_timeout = default_timeout
        self.re_challenge_cooldown = re_challenge_cooldown

        # Reputation decay rates per penalty level (from Web4 Session #84)
        self.decay_rates = decay_rates or {
            EvasionPenaltyLevel.NONE: 0.0,        # No decay
            EvasionPenaltyLevel.WARNING: 0.05,    # 5% decay
            EvasionPenaltyLevel.MODERATE: 0.15,   # 15% decay
            EvasionPenaltyLevel.SEVERE: 0.30,     # 30% decay
            EvasionPenaltyLevel.PERMANENT: 0.50,  # 50% decay (permanent)
        }

        # Challenge tracking
        self.challenges: Dict[str, QualityChallenge] = {}

        # Platform evasion records
        self.evasion_records: Dict[str, EvasionRecord] = {}

        # Challenge history: platform_lct_id → [challenge_ids]
        self.platform_challenges: Dict[str, List[str]] = defaultdict(list)

        # Statistics
        self.total_challenges_issued: int = 0
        self.total_responses: int = 0
        self.total_evasions: int = 0
        self.total_penalties_applied: int = 0

    def get_evasion_record(self, platform_lct_id: str) -> EvasionRecord:
        """Get or create evasion record for platform"""
        if platform_lct_id not in self.evasion_records:
            self.evasion_records[platform_lct_id] = EvasionRecord(
                platform_lct_id=platform_lct_id
            )
        return self.evasion_records[platform_lct_id]

    def can_challenge(self,
                      platform_lct_id: str,
                      current_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if platform can be challenged

        Args:
            platform_lct_id: Platform to potentially challenge
            current_time: Current timestamp

        Returns:
            (can_challenge, reason)
        """
        current_time = current_time or time.time()

        # Check for recent challenges (cooldown to prevent spam)
        record = self.get_evasion_record(platform_lct_id)

        if record.last_challenge is not None:
            time_since_last = current_time - record.last_challenge

            if time_since_last < self.re_challenge_cooldown:
                remaining = self.re_challenge_cooldown - time_since_last
                return False, f"cooldown: {remaining:.0f}s remaining"

        return True, "allowed"

    def issue_challenge(self,
                        platform_lct_id: str,
                        challenger_lct_id: str,
                        execution_proof: ExecutionProof,
                        timeout_period: Optional[float] = None) -> Tuple[bool, str, Optional[QualityChallenge]]:
        """
        Issue quality challenge to platform

        Args:
            platform_lct_id: Platform being challenged
            challenger_lct_id: Platform/witness issuing challenge
            execution_proof: ExecutionProof being challenged
            timeout_period: Custom timeout (or use default)

        Returns:
            (success, reason, challenge)
        """
        # Check if can challenge
        can_challenge, reason = self.can_challenge(platform_lct_id)
        if not can_challenge:
            return False, reason, None

        # Create challenge
        challenge_id = f"challenge_{platform_lct_id}_{int(time.time() * 1000)}"
        timeout = timeout_period or self.default_timeout

        challenge = QualityChallenge(
            challenge_id=challenge_id,
            platform_lct_id=platform_lct_id,
            challenger_lct_id=challenger_lct_id,
            challenged_proof_id=execution_proof.task_id,
            claimed_quality=execution_proof.quality_score,
            timeout_period=timeout
        )

        # Register challenge
        self.challenges[challenge_id] = challenge
        self.platform_challenges[platform_lct_id].append(challenge_id)
        self.total_challenges_issued += 1

        # Update evasion record to track last challenge time (for cooldown)
        record = self.get_evasion_record(platform_lct_id)
        if record.first_challenge is None:
            record.first_challenge = challenge.issue_timestamp
        record.last_challenge = challenge.issue_timestamp

        return True, "challenge_issued", challenge

    def respond_to_challenge(self,
                            challenge_id: str,
                            evidence: Dict[str, Any],
                            current_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Platform responds to challenge

        Args:
            challenge_id: Challenge to respond to
            evidence: Evidence supporting quality claim
            current_time: Current timestamp

        Returns:
            (success, reason)
        """
        if challenge_id not in self.challenges:
            return False, "challenge_not_found"

        challenge = self.challenges[challenge_id]

        if challenge.respond(evidence, current_time):
            self.total_responses += 1
            return True, "response_accepted"
        else:
            return False, "timeout_expired"

    def verify_challenge_response(self,
                                  challenge_id: str,
                                  verified_quality: float,
                                  is_valid: bool) -> Tuple[bool, str]:
        """
        Verify platform's challenge response

        Args:
            challenge_id: Challenge being verified
            verified_quality: Measured quality from verification
            is_valid: Whether response was valid

        Returns:
            (success, reason)
        """
        if challenge_id not in self.challenges:
            return False, "challenge_not_found"

        challenge = self.challenges[challenge_id]
        challenge.verify_response(verified_quality, is_valid)

        # Update evasion record
        record = self.get_evasion_record(challenge.platform_lct_id)
        record.add_challenge(challenge)

        return True, "response_verified"

    def check_timeouts(self, current_time: Optional[float] = None) -> List[QualityChallenge]:
        """
        Check for timed-out challenges and mark as evaded

        Args:
            current_time: Current timestamp

        Returns:
            List of newly evaded challenges
        """
        current_time = current_time or time.time()
        evaded = []

        for challenge in self.challenges.values():
            if challenge.has_timed_out(current_time):
                challenge.mark_evaded()

                # Update evasion record
                record = self.get_evasion_record(challenge.platform_lct_id)
                record.add_challenge(challenge)

                self.total_evasions += 1
                evaded.append(challenge)

        return evaded

    def apply_evasion_penalty(self,
                             platform: FederationIdentity,
                             current_time: Optional[float] = None) -> float:
        """
        Calculate and apply evasion penalty to platform reputation

        Args:
            platform: Platform to penalize
            current_time: Current timestamp

        Returns:
            New reputation score after penalty
        """
        # Check for timeouts first
        self.check_timeouts(current_time)

        # Get evasion record
        record = self.get_evasion_record(platform.lct_id)

        # Determine penalty level
        penalty_level = record.get_penalty_level()
        decay_rate = self.decay_rates[penalty_level]

        if decay_rate > 0:
            # Apply decay to reputation
            original_rep = platform.reputation_score
            platform.reputation_score *= (1.0 - decay_rate)
            self.total_penalties_applied += 1

            return platform.reputation_score

        return platform.reputation_score

    def get_platform_challenge_stats(self, platform_lct_id: str) -> Dict[str, Any]:
        """
        Get challenge statistics for platform

        Args:
            platform_lct_id: Platform to get stats for

        Returns:
            Dictionary with challenge statistics
        """
        record = self.get_evasion_record(platform_lct_id)
        penalty_level = record.get_penalty_level()

        return {
            "total_challenges": record.total_challenges,
            "responded": record.responded_challenges,
            "evaded": record.evaded_challenges,
            "response_rate": record.get_response_rate(),
            "evasion_rate": record.get_evasion_rate(),
            "strikes": record.strike_count,
            "penalty_level": penalty_level.name,
            "decay_rate": self.decay_rates[penalty_level],
            "average_verified_quality": record.average_verified_quality,
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        return {
            "total_challenges": self.total_challenges_issued,
            "total_responses": self.total_responses,
            "total_evasions": self.total_evasions,
            "total_penalties": self.total_penalties_applied,
            "response_rate": (self.total_responses / self.total_challenges_issued
                            if self.total_challenges_issued > 0 else 0.0),
            "evasion_rate": (self.total_evasions / self.total_challenges_issued
                           if self.total_challenges_issued > 0 else 0.0),
            "platforms_tracked": len(self.evasion_records),
        }
