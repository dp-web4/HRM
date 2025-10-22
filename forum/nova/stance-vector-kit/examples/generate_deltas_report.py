
import argparse, os
from stancekit.deltas import compute_deltas, render_delta_html

HTML_WRAPPER_HEAD = "<html><head><meta charset='utf-8'><style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0;} table.tbl{border-collapse:collapse;width:100%;} table.tbl th, table.tbl td{border:1px solid #eee;padding:6px 8px;text-align:left;}</style></head><body>"
HTML_WRAPPER_TAIL = "</body></html>"

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs_dirs', nargs='+', required=True, help='session output dirs (first is baseline)')
    ap.add_argument('--session_names', nargs='*', default=None, help='optional names for sessions')
    ap.add_argument('--keys', nargs='*', default=None, help='metric keys to diff (default: cross_context_cosine flicker_index)')
    ap.add_argument('--output_html', required=True)
    args = ap.parse_args()

    res = compute_deltas(args.inputs_dirs, keys=args.keys)
    html = render_delta_html(args.session_names or [], res)
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    with open(args.output_html, 'w', encoding='utf-8') as f:
        f.write(HTML_WRAPPER_HEAD + html + HTML_WRAPPER_TAIL)
    print('Wrote', args.output_html)
