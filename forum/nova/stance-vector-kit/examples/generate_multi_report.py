
import argparse
from stancekit.report_multi import generate_multi_report

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs_dirs', nargs='+', required=True, help='space-separated list of session output dirs')
    ap.add_argument('--session_names', nargs='*', default=None, help='optional names for sessions, same order as inputs')
    ap.add_argument('--output_html', required=True)
    ap.add_argument('--title', default='SVK Multi-Session Report')
    args = ap.parse_args()
    out = generate_multi_report(args.inputs_dirs, args.session_names, args.output_html, title=args.title)
    print('Wrote', out)
