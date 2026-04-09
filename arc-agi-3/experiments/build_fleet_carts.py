#!/usr/bin/env python3
"""
Build fleet learning brain carts from all available knowledge.

Converts markdown knowledge docs (world models, meta-lessons, mechanics)
into JSONL entries compatible with federate.py, then runs migration to
build brain carts.

This is the bridge between human-readable fleet learning docs and the
semantic search substrate (membot brain carts).

Usage:
    python build_fleet_carts.py [--dry-run]
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime

FLEET_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'shared-context', 'arc-agi-3', 'fleet-learning')

# Add membot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'membot'))


def extract_game_prefix(filename):
    """Extract game prefix from filename like 'cd82_world_model.md'."""
    match = re.match(r'^([a-z0-9]+)_', filename)
    return match.group(1) if match else None


def split_markdown_sections(text):
    """Split markdown into sections by ## headers."""
    sections = []
    current_header = None
    current_body = []

    for line in text.split('\n'):
        if line.startswith('## '):
            if current_header is not None:
                sections.append((current_header, '\n'.join(current_body).strip()))
            current_header = line[3:].strip()
            current_body = []
        elif line.startswith('# ') and current_header is None:
            # Top-level header — use as context but don't create section
            current_body.append(line)
        else:
            current_body.append(line)

    if current_header is not None:
        sections.append((current_header, '\n'.join(current_body).strip()))
    elif current_body:
        sections.append(("Overview", '\n'.join(current_body).strip()))

    return sections


def md_to_entries(filepath, machine_id):
    """Convert a markdown file into JSONL entries for cart embedding."""
    filename = os.path.basename(filepath)
    game = extract_game_prefix(filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract title from first line
    title_match = re.match(r'^#\s+(.+)', text)
    title = title_match.group(1) if title_match else filename

    # Extract date if present
    date_match = re.search(r'\*(\d{4}-\d{2}-\d{2})', text)
    timestamp = date_match.group(1) + "T00:00:00" if date_match else "2026-04-09T00:00:00"

    entries = []

    # Determine document type from filename
    if 'world_model' in filename:
        doc_type = 'world_model'
    elif 'mechanics' in filename:
        doc_type = 'mechanics'
    elif 'complete' in filename:
        doc_type = 'game_complete_analysis'
    elif 'meta_' in filename:
        doc_type = 'meta_learning'
        game = '_meta'  # game-agnostic
    else:
        doc_type = 'knowledge'

    sections = split_markdown_sections(text)

    for header, body in sections:
        if not body or len(body) < 20:
            continue

        # Skip metadata/title sections
        if header.lower() in ('game type', 'level progression'):
            # These are useful — keep them
            pass

        # Determine event type based on content
        if any(kw in header.lower() for kw in ['lesson', 'meta', 'principle', 'insight']):
            event = 'meta_learning'
        elif any(kw in header.lower() for kw in ['mechanic', 'physics', 'model']):
            event = 'mechanic_discovered'
        elif any(kw in header.lower() for kw in ['solution', 'solved', 'summary']):
            event = 'solution_summary'
        elif any(kw in header.lower() for kw in ['obstacle', 'animation', 'feedback']):
            event = 'mechanic_discovered'
        elif any(kw in header.lower() for kw in ['question', 'open']):
            event = 'open_question'
        elif any(kw in header.lower() for kw in ['update', 'discovery', 'learning']):
            event = 'discovery'
        elif any(kw in header.lower() for kw in ['protocol', 'assessment', 'discipline']):
            event = 'strategy'
        else:
            event = 'knowledge'

        # Build the entry
        entry = {
            "timestamp": timestamp,
            "machine": machine_id,
            "player": "claude",
            "game": game if game != '_meta' else "all",
            "level": -1,
            "event": event,
            "description": f"[{doc_type}] {header}: {body[:500]}",
            "structural_pattern": doc_type,
            "meta": f"From {filename}: {header}",
            "confidence": "high" if 'verified' in body.lower() or 'confirmed' in body.lower() else "medium",
            "_source_machine": machine_id,
            "_source_file": filename,
            "_source_section": header,
        }

        # For mechanic entries, extract the rule
        if event == 'mechanic_discovered':
            entry["mechanic"] = header.lower().replace(' ', '_')
            entry["description"] = body[:1000]

        # For meta-learning, make it game-agnostic and prominent
        if event == 'meta_learning' or doc_type == 'meta_learning':
            entry["game"] = "all"
            entry["insight"] = header
            entry["description"] = body[:1000]

        # For solution summaries, extract actions if present
        if event == 'solution_summary':
            actions_match = re.search(r'(\d+)\s*steps', body)
            if actions_match:
                entry["actions"] = int(actions_match.group(1))

        entries.append(entry)

    # If doc is small enough, also add the entire document as one entry
    # (captures cross-section relationships that per-section entries miss)
    if len(text) < 4000:
        entries.append({
            "timestamp": timestamp,
            "machine": machine_id,
            "player": "claude",
            "game": game if game != '_meta' else "all",
            "level": -1,
            "event": "complete_document",
            "description": text[:3000],
            "structural_pattern": doc_type,
            "meta": f"Full document: {title}",
            "_source_machine": machine_id,
            "_source_file": filename,
        })

    return entries


def find_markdown_files(fleet_dir):
    """Find all markdown files in fleet learning directory."""
    md_files = []
    for machine_id in os.listdir(fleet_dir):
        machine_dir = os.path.join(fleet_dir, machine_id)
        if not os.path.isdir(machine_dir):
            continue
        for f in os.listdir(machine_dir):
            if f.endswith('.md'):
                md_files.append((machine_id, os.path.join(machine_dir, f)))
    return md_files


def main():
    dry_run = '--dry-run' in sys.argv

    fleet_dir = os.path.abspath(FLEET_DIR)
    if not os.path.exists(fleet_dir):
        print(f"Fleet dir not found: {fleet_dir}")
        sys.exit(1)

    print(f"Fleet directory: {fleet_dir}")
    print()

    # Step 1: Convert markdown files to JSONL entries
    md_files = find_markdown_files(fleet_dir)
    print(f"Found {len(md_files)} markdown files to process:")

    all_new_entries = {}  # machine_id -> [entries]

    for machine_id, filepath in md_files:
        filename = os.path.basename(filepath)
        entries = md_to_entries(filepath, machine_id)
        print(f"  {machine_id}/{filename}: {len(entries)} entries")

        if machine_id not in all_new_entries:
            all_new_entries[machine_id] = []
        all_new_entries[machine_id].extend(entries)

    print()

    # Step 2: Write new entries to *_knowledge.jsonl files
    # (separate from *_learning.jsonl to keep them distinct)
    for machine_id, entries in all_new_entries.items():
        knowledge_file = os.path.join(fleet_dir, machine_id, 'knowledge_from_docs.jsonl')

        if dry_run:
            print(f"[DRY RUN] Would write {len(entries)} entries to {knowledge_file}")
            for e in entries[:3]:
                print(f"  - [{e['event']}] {e.get('_source_section', e.get('meta', ''))[:80]}")
            if len(entries) > 3:
                print(f"  ... and {len(entries) - 3} more")
            continue

        # Write the knowledge JSONL
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Wrote {len(entries)} entries to {knowledge_file}")

    if dry_run:
        print("\n[DRY RUN] No files written. Remove --dry-run to execute.")
        return

    # Step 3: Run federate.migrate_jsonl() to rebuild carts
    print("\n--- Running federate.migrate_jsonl() ---")
    try:
        import federate
        # The migrate function looks for *_learning.jsonl only.
        # We need to also handle our *_knowledge.jsonl files.
        # Approach: temporarily rename, or call publish_jsonl_lines directly.

        for machine_id, entries in all_new_entries.items():
            print(f"\nPublishing {len(entries)} knowledge entries for {machine_id}...")
            result = federate.publish_jsonl_lines(entries, machine_id, fleet_dir)
            print(f"  Added: {result.get('added', 0)}, "
                  f"Skipped (dedup): {result.get('skipped', 0)}, "
                  f"Cart: {result.get('cart_path', '?')}")

        # Also re-migrate existing JSONL learning files (in case they've been updated)
        print("\nRe-migrating existing learning JSONL files...")
        migrate_result = federate.migrate_jsonl(fleet_dir, in_place=True)
        print(f"  Machines: {migrate_result['machines_processed']}, "
              f"Entries: {migrate_result['total_entries']}, "
              f"Carts: {migrate_result['carts_built']}")
        if migrate_result.get('errors'):
            print(f"  Errors: {migrate_result['errors']}")

        # Step 4: Load fleet to verify
        print("\n--- Verifying fleet load ---")
        load_result = federate.load_fleet(fleet_dir)
        print(f"  Mounted: {load_result.get('mounted', 0)} carts")
        print(f"  Total patterns: {load_result.get('total_patterns', 0)}")
        print(f"  Machines: {load_result.get('machines', [])}")

        # Step 5: Test a semantic query
        import multi_cart
        print("\n--- Test queries ---")
        test_queries = [
            "how to navigate around obstacles in a walking puzzle",
            "predict before acting, world model in context",
            "how to select the right endpoint when compasses overlap",
            "efficiency: see delta between current state and target",
            "getting unstuck when repeating the same action",
        ]
        for q in test_queries:
            results = multi_cart.search(q, top_k=3, scope="all",
                                        role_filter="federated")
            hits = results.get("results", [])
            print(f"\n  Q: \"{q}\"")
            for h in hits[:2]:
                score = h.get("score", 0)
                cart = h.get("cart_id", "?")
                text_preview = h.get("text", "")[:100].replace('\n', ' ')
                print(f"    [{cart}] {score:.3f}: {text_preview}")

    except ImportError as e:
        print(f"\nCannot import federate/multi_cart: {e}")
        print("Knowledge JSONL files were written. Run federate.migrate_jsonl() manually.")
    except Exception as e:
        print(f"\nError during cart building: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
