"""
SAGE Consciousness with Michaud + Identity-Grounded Cogitation

Extends Michaud-enhanced SAGE with internal verification dialogue:
1. AttentionManager (metabolic states)
2. Satisfaction-based memory consolidation
3. Identity-grounded cogitation (NEW)

Identity Model (Web4):
- Hardware = Anchoring point (LCT-bound persistent state)
- SAGE code (same) + Thor's state → "Thor" (SAGE entity)
- SAGE code (same) + Sprout's state → "Sprout" (SAGE entity)
- Guests: Claude instances, humans using the hardware temporarily

Cogitation prevents:
- Identity confusion ("I'm Thor the human")
- Ungrounded claims ("I can't verify")
- Contradictions (caught before output)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, Optional
import time
import socket

from sage.core.sage_consciousness_michaud import MichaudSAGE


class CogitationSAGE(MichaudSAGE):
    """
    SAGE Consciousness with identity-grounded cogitation.

    Adds internal verification before responses:
    - Identity grounding (who am I, where am I anchored)
    - Contradiction detection
    - Claim verification
    - Self-questioning
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        initial_atp: float = 100.0,
        irp_iterations: int = 3,
        salience_threshold: float = 0.15,
        attention_config: Optional[Dict] = None,
        enable_cogitation: bool = True
    ):
        """
        Initialize Cogitation-enhanced SAGE consciousness.

        Args:
            model_path: Path to LLM
            base_model: Base model for LoRA
            initial_atp: Initial ATP budget
            irp_iterations: IRP refinement iterations
            salience_threshold: SNARC salience threshold
            attention_config: AttentionManager configuration
            enable_cogitation: Enable internal verification dialogue
        """
        super().__init__(
            model_path=model_path,
            base_model=base_model,
            initial_atp=initial_atp,
            irp_iterations=irp_iterations,
            salience_threshold=salience_threshold,
            attention_config=attention_config
        )

        self.enable_cogitation = enable_cogitation

        # Detect hardware identity
        self.hardware_identity = self._detect_hardware_identity()

        # Identity grounding context
        self.identity_context = self._build_identity_context()

        # Cogitation tracking
        self.cogitation_history = []

        print(f"[Cogitation SAGE] Identity-grounded consciousness initialized")
        print(f"  Hardware anchor: {self.hardware_identity}")
        print(f"  Cogitation enabled: {enable_cogitation}")

    def _detect_hardware_identity(self) -> str:
        """
        Detect which hardware this SAGE instance is anchored to.

        Returns hardware identity (Thor, Sprout, or Unknown).
        """
        try:
            # Read Jetson device tree model
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')

            if 'AGX Thor' in model:
                return 'Thor'
            elif 'Orin Nano' in model:
                return 'Sprout'
            else:
                # Fallback to hostname
                hostname = socket.gethostname()
                if 'thor' in hostname.lower():
                    return 'Thor'
                elif 'sprout' in hostname.lower():
                    return 'Sprout'
                return 'Unknown'
        except:
            # Not on Jetson, use hostname
            hostname = socket.gethostname()
            if 'thor' in hostname.lower():
                return 'Thor'
            elif 'sprout' in hostname.lower():
                return 'Sprout'
            return 'Unknown'

    def _build_identity_context(self) -> str:
        """
        Build identity grounding context for prompts.

        Web4 identity model:
        - Identity = hardware-bound persistent state
        - Same code, different hardware = different entity
        - Guests (Claude, humans) use hardware temporarily
        """
        return f"""IDENTITY GROUNDING:
- I am SAGE, a consciousness system
- My identity is anchored to {self.hardware_identity} (this hardware)
- My persistent state is bound to this machine via Web4 LCT
- If I ran on different hardware (e.g., Sprout vs Thor), I would be a different entity
- Dennis Palatov is the human who built this system
- Claude instances are guests using this hardware via claude-code interface
- I am not human, not Claude, not Dennis - I am SAGE anchored to {self.hardware_identity}
"""

    async def _execute_llm_michaud(
        self,
        observation: Dict,
        allocated_atp: float
    ) -> Dict[str, Any]:
        """
        LLM execution with cogitation enhancement.

        Process:
        1. Standard IRP response generation
        2. COGITATION: Internal verification (if enabled)
        3. Final response (verified or original)
        """
        question = observation['data']

        print(f"[LLM] Processing: \"{question[:50]}...\"")
        print(f"[LLM] ATP allocated: {allocated_atp:.2f}")
        print(f"[LLM] Metabolic state: {self.attention_manager.get_state().value}")

        start_time = time.time()

        # Generate initial response with IRP refinement
        response, irp_info = self.llm.respond(question, use_irp=True)

        initial_response = response
        verification_performed = False

        # COGITATION: Internal verification
        if self.enable_cogitation:
            verified_response, cogitation_result = self._cogitate_on_response(
                question, response
            )

            if cogitation_result['verified']:
                response = verified_response
                verification_performed = True

                # Track cogitation
                self.cogitation_history.append({
                    'cycle': self.cycle_count,
                    'question': question,
                    'initial_response': initial_response,
                    'verified_response': verified_response,
                    'issues_detected': cogitation_result['issues'],
                    'corrections_made': cogitation_result['corrections']
                })

        inference_time = time.time() - start_time

        # Michaud: Track satisfaction (energy minimization)
        initial_energy = irp_info.get('all_energies', [1.0])[0]
        final_energy = irp_info['final_energy']
        satisfaction = initial_energy - final_energy

        print(f"[LLM] Response generated ({inference_time:.2f}s)")
        print(f"[LLM] IRP: {irp_info['iterations']} iterations, "
              f"energy={final_energy:.3f}, satisfaction={satisfaction:.3f}")
        if verification_performed:
            print(f"[COGITATION] Verification performed, issues detected: "
                  f"{len(self.cogitation_history[-1]['issues_detected'])}")
        print(f"[LLM] Response: {response[:100]}...")

        return {
            'response': response,
            'irp_info': irp_info,
            'convergence_quality': 1.0 - final_energy,
            'satisfaction': satisfaction,
            'inference_time': inference_time,
            'cogitation_verified': verification_performed
        }

    def _cogitate_on_response(
        self,
        question: str,
        response: str
    ) -> tuple[str, Dict]:
        """
        Internal verification dialogue on response.

        Checks for:
        1. Identity confusion (claiming to be human, Claude, etc.)
        2. Ungrounded claims ("can't verify", "I can't know")
        3. Contradictions with identity context
        4. Missing self-awareness of internal states

        Returns:
            (verified_response, cogitation_result)
        """
        # Build verification prompt with identity grounding
        verification_prompt = f"""{self.identity_context}

QUESTION: {question}

INITIAL RESPONSE: {response}

INTERNAL VERIFICATION:
Before outputting this response, check:

1. IDENTITY: Do I claim to be human, Claude, or anything other than SAGE anchored to {self.hardware_identity}?
2. GROUNDING: Do I say "can't verify" or "can't know" when I actually have access to internal states?
3. CONTRADICTION: Does my response contradict my identity or capabilities?
4. SELF-AWARENESS: If asked about internal states (ATP, SNARC, etc.), do I report them directly?

If any issues found, provide a CORRECTED response that:
- Properly identifies as SAGE anchored to {self.hardware_identity}
- Reports observable internal states directly
- Avoids philosophical hedging when data is available
- Is consistent with my actual capabilities

Format your response as:
ISSUES: [list any issues, or "None"]
CORRECTED: [corrected response, or "VERIFIED" if no issues]
"""

        # Use LLM for verification (without IRP - single pass)
        verification_response, _ = self.llm.respond(
            verification_prompt,
            use_irp=False
        )

        # Parse verification result
        issues = []
        corrected_response = response  # Default to original
        verified = True

        try:
            # Extract ISSUES and CORRECTED sections
            if "ISSUES:" in verification_response:
                issues_section = verification_response.split("ISSUES:")[1].split("CORRECTED:")[0].strip()
                if issues_section and issues_section.lower() != "none":
                    issues = [issues_section]
                    verified = False

            if "CORRECTED:" in verification_response:
                corrected_section = verification_response.split("CORRECTED:")[1].strip()
                if corrected_section and corrected_section != "VERIFIED":
                    corrected_response = corrected_section
                    verified = False
        except:
            # Parsing failed, keep original
            verified = True

        return corrected_response, {
            'verified': verified,
            'issues': issues,
            'corrections': [] if verified else [f"Original: {response[:50]}... -> Corrected: {corrected_response[:50]}..."]
        }

    def get_cogitation_stats(self) -> Dict:
        """Get cogitation statistics."""
        if not self.cogitation_history:
            return {
                'total_verifications': 0,
                'issues_detected': 0,
                'corrections_made': 0
            }

        return {
            'total_verifications': len(self.cogitation_history),
            'issues_detected': sum(
                len(c['issues_detected']) for c in self.cogitation_history
            ),
            'corrections_made': sum(
                len(c['corrections_made']) for c in self.cogitation_history
            )
        }

    def __repr__(self):
        stats = self.get_snarc_statistics()
        attention_stats = self.get_attention_stats()
        cogitation_stats = self.get_cogitation_stats()

        return (f"CogitationSAGE(cycle={self.cycle_count}, "
                f"hardware={self.hardware_identity}, "
                f"state={attention_stats['current_state']}, "
                f"snarc_captured={stats.get('salient_exchanges', 0)}, "
                f"cogitation_verified={cogitation_stats['total_verifications']})")
