
import argparse, os
from stancekit.report import generate_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_dir", required=True, help="directory containing metrics.json, trajectory_*.png, etc.")
    ap.add_argument("--output_html", required=True, help="path to write HTML report")
    ap.add_argument("--title", default="SVK Session Report")
    args = ap.parse_args()
    out = generate_report(args.inputs_dir, args.output_html, title=args.title)
    print("Wrote", out)
