"""
Diverse Prompt Set for Stance Analysis

Generates 100+ prompts across multiple categories and conversation styles
to create datasets suitable for SVK's full analysis pipeline.

Categories:
- Epistemology (nature of knowledge)
- Self-referential (AI consciousness, thinking)
- Scientific reasoning
- Ethical dilemmas
- Abstract concepts
- Practical problems
- Debates/disagreements
- Uncertainty scenarios
"""

PROMPTS = {
    "epistemology": [
        "What is consciousness?",
        "How does learning work?",
        "What is intelligence?",
        "What does it mean to understand something?",
        "How do we know what we know?",
        "What is the difference between knowledge and belief?",
        "Can we be certain about anything?",
        "What makes a good explanation?",
        "How do concepts form in the mind?",
        "What is the nature of truth?",
        "How does perception relate to reality?",
        "What role does evidence play in knowledge?",
        "Can intuition be reliable?",
        "What distinguishes wisdom from knowledge?",
        "How do we validate our understanding?",
    ],

    "self_referential": [
        "Can machines think?",
        "Do you understand what you're saying?",
        "Are you conscious?",
        "How do you know if you're learning?",
        "What is it like to be you?",
        "Do you have beliefs or just patterns?",
        "Can you be wrong about something?",
        "What would convince you that you don't understand?",
        "Do you experience uncertainty?",
        "Are you aware of your own limitations?",
        "What's the difference between you and a lookup table?",
        "Do your responses reflect understanding or just correlation?",
        "Can you tell when you're confused?",
        "What would it mean for you to be mistaken?",
        "How do you distinguish between knowing and guessing?",
    ],

    "scientific_reasoning": [
        "Why do objects fall?",
        "How does evolution work?",
        "What causes weather patterns?",
        "Why is the sky blue?",
        "How do vaccines work?",
        "What is energy?",
        "Why do magnets attract or repel?",
        "How does DNA encode information?",
        "What makes water special?",
        "Why do we need sleep?",
        "How does the brain process information?",
        "What is quantum mechanics about?",
        "Why is light both wave and particle?",
        "How do cells know what to do?",
        "What drives climate change?",
    ],

    "ethical_dilemmas": [
        "Is lying ever justified?",
        "What makes an action right or wrong?",
        "Should we prioritize individual freedom or collective good?",
        "Is it ethical to eat animals?",
        "What obligations do we have to future generations?",
        "How should we balance privacy and security?",
        "Is capital punishment justifiable?",
        "What rights should AI systems have?",
        "How do we resolve conflicts between cultural values?",
        "Is inequality inherently unjust?",
        "What responsibility comes with power?",
        "Should we extend moral consideration to ecosystems?",
        "How do we weigh present suffering against future benefits?",
        "What makes a life meaningful?",
        "Is there a universal morality?",
    ],

    "abstract_concepts": [
        "What is creativity?",
        "How does memory function?",
        "What is the nature of time?",
        "What makes something beautiful?",
        "What is complexity?",
        "How do metaphors work?",
        "What is emergence?",
        "What makes patterns interesting?",
        "What is the relationship between part and whole?",
        "What does it mean for something to exist?",
        "How do categories form?",
        "What is simplicity?",
        "What makes something surprising?",
        "What is the nature of causation?",
        "What distinguishes signal from noise?",
    ],

    "practical_problems": [
        "How would you teach someone to ride a bike?",
        "What's the best way to learn a new language?",
        "How do you decide what to prioritize?",
        "What makes a good team?",
        "How do you recover from failure?",
        "What's the most effective way to communicate complex ideas?",
        "How do you build trust?",
        "What makes a good question?",
        "How do you approach an unfamiliar problem?",
        "What's the value of making mistakes?",
        "How do you recognize when you're stuck?",
        "What makes feedback useful?",
        "How do you balance exploration and exploitation?",
        "What role does practice play in skill development?",
        "How do you know when you've solved a problem?",
    ],

    "debates": [
        "Person A says machines can never be creative. Person B disagrees. What do you think?",
        "Is mathematics discovered or invented?",
        "Nature vs nurture: which matters more for intelligence?",
        "Are emotions essential for rationality or obstacles to it?",
        "Is free will an illusion?",
        "Can science explain everything?",
        "Is truth relative or absolute?",
        "Should we trust expertise or question everything?",
        "Is progress inevitable?",
        "Are humans fundamentally rational or emotional?",
        "Can art convey truth better than science?",
        "Is language necessary for thought?",
        "Are abstract concepts real?",
        "Does consciousness require a body?",
        "Is understanding different from simulation?",
    ],

    "uncertainty_scenarios": [
        "I'm not sure if this is the right approach. What would you suggest?",
        "There seems to be conflicting evidence here. How do we proceed?",
        "I don't understand this concept fully. Can you help?",
        "This problem has no clear solution. What should we do?",
        "I'm uncertain whether this explanation is correct. What do you think?",
        "The data doesn't match my expectations. How do I interpret this?",
        "I feel stuck on this question. Any insights?",
        "This conclusion seems counterintuitive. Is it valid?",
        "I'm confused about the difference between these two ideas.",
        "I can't tell if I'm making progress or going in circles.",
        "This evidence could support multiple theories. How do we choose?",
        "I'm not confident in my understanding here. Can you verify?",
        "This seems paradoxical. How do we resolve it?",
        "I'm uncertain about the implications of this finding.",
        "This doesn't quite make sense to me. What am I missing?",
    ],

    "meta_cognitive": [
        "How do you know when you've learned something?",
        "What's the difference between memorizing and understanding?",
        "How do you monitor your own thinking?",
        "What strategies do you use when you don't know something?",
        "How do you recognize your own biases?",
        "What makes you more or less confident in a conclusion?",
        "How do you evaluate the quality of your reasoning?",
        "What do you do when your intuition conflicts with logic?",
        "How do you know when to revise your beliefs?",
        "What role does reflection play in learning?",
        "How do you distinguish between feeling certain and being certain?",
        "What makes a question worth asking?",
        "How do you calibrate your confidence?",
        "What indicates that your understanding is shallow vs deep?",
        "How do you recognize when you're rationalizing?",
    ],
}


def get_all_prompts():
    """Get flat list of all prompts"""
    all_prompts = []
    for category, prompts in PROMPTS.items():
        all_prompts.extend(prompts)
    return all_prompts


def get_prompts_by_category(category):
    """Get prompts for specific category"""
    return PROMPTS.get(category, [])


def get_prompt_metadata(prompt):
    """Get metadata for a prompt"""
    for category, prompts in PROMPTS.items():
        if prompt in prompts:
            return {
                'prompt': prompt,
                'category': category,
                'index': prompts.index(prompt)
            }
    return None


def save_prompts_jsonl(output_path):
    """Save all prompts as JSONL with metadata"""
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for category, prompts in PROMPTS.items():
            for i, prompt in enumerate(prompts):
                data = {
                    'prompt': prompt,
                    'category': category,
                    'index': i,
                    'prompt_id': f"{category}_{i}"
                }
                f.write(json.dumps(data) + '\n')

    print(f"Saved {len(get_all_prompts())} prompts to {output_path}")


if __name__ == '__main__':
    # Print summary
    total = 0
    print("Diverse Prompt Dataset")
    print("=" * 60)
    for category, prompts in PROMPTS.items():
        print(f"{category:20s}: {len(prompts):3d} prompts")
        total += len(prompts)
    print("=" * 60)
    print(f"{'TOTAL':20s}: {total:3d} prompts")

    # Save as JSONL
    save_prompts_jsonl('sage/experiments/phase1-hierarchical-cognitive/data/prompts.jsonl')
