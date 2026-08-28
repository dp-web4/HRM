# Collection safety for `sage/tests` (audit #31).
#
# pytest IMPORTS every test_*.py at collection time. Several files here are not
# pytest tests at all — they are manual scripts that call `from_pretrained` (loading
# multi-GB / 30B models) and `sys.exit()` at module top level, with no
# `if __name__ == "__main__"` guard. A bare `pytest` would try to load a model on
# collection and/or abort the whole session via SystemExit. These are skipped at
# collection here until they are either guarded or renamed off the test_ prefix.

collect_ignore = [
    # Acute: import-time model load + sys.exit — MUST ignore.
    "test_full_huggingface_model.py",
    "test_full_model_direct_import.py",
    "test_full_model_proper_class.py",
    "test_full_model_with_swap.py",
    "test_native_q3_omni.py",
    "test_official_inference_pattern.py",
    "test_vllm_q3_omni.py",
    # Manual scripts (no pytest test functions) that execute real logic on import.
    "test_full_model.py",
    "test_cognitive_simulated.py",
    "test_complexity_aware_iterations.py",
    "test_generation_simple.py",
    "test_greedy.py",
    "test_hybrid_learning.py",
    "test_mrope_fix.py",
    "test_nemotron_vs_q3omni.py",
    "test_per_token_debug.py",
    "test_power_profile_edge.py",
    "test_quality_validation.py",
    "test_quality_validation_extended.py",
    "test_single_expert_output.py",
    "test_thermal_stability_edge.py",
    "test_trust_based_generation_integration.py",
    "test_trust_first_comparison.py",
    "test_atp_metering_edge.py",
    "test_mrh_full_pipeline_edge.py",
    "test_sage_web4_bridge_edge.py",
    "test_voice_conversation_latency.py",
    "test_streaming_audio.py",
    "test_simple_audio.py",
    "test_conversation.py",
    "test_mrh_debug.py",
    "test_sage_rev0_extended.py",
    "test_sage_rev1_circadian.py",
    "test_cross_modal_attention.py",
    "test_qwen3_omni.py",
]
