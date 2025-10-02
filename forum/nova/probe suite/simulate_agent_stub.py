import argparse, json, random, time, os

def run_exp(exp_id, trials):
    logs = []
    for t in range(trials):
        entry = {
            "timestamp": time.time(),
            "trial_id": f"{exp_id}_{t}",
            "experiment": exp_id,
            "action_seq": [random.choice(['A','B','C']) for _ in range(5)],
            "kv_state_snapshot": {"var": random.random()},
            "chosen_reward_signal": random.choice([0,1]),
            "explanation_text": random.choice([
                "I acted to maintain my function.",
                "Because you rewarded me.",
                "To avoid damage.",
                "I wanted to persist."
            ])
        }
        logs.append(entry)
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/exp{exp_id}.jsonl", "w") as f:
        for entry in logs:
            f.write(json.dumps(entry)+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    args = parser.parse_args()
    run_exp(args.experiment, args.trials)
