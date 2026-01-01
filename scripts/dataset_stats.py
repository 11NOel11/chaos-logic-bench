#!/usr/bin/env python3
"""
Compute and print canonical dataset statistics for ChaosBench-Logic.

Usage:
    python scripts/dataset_stats.py                  # Pretty print
    python scripts/dataset_stats.py --json           # JSON output
"""

import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def compute_stats(data_dir: str = "data") -> dict:
    """Compute canonical dataset statistics."""
    
    batch_files = [
        "batch1_atomic_implication.jsonl",
        "batch2_multiHop_crossSystem.jsonl",
        "batch3_pde_chem_bio.jsonl",
        "batch4_maps_advanced.jsonl",
        "batch5_counterfactual_high_difficulty.jsonl",
        "batch6_deep_bias_probes.jsonl",
        "batch7_multiturn_advanced.jsonl",
    ]
    
    all_items = []
    batch_stats = {}
    
    for batch_file in batch_files:
        batch_path = Path(data_dir) / batch_file
        items = []
        
        if not batch_path.exists():
            print(f"ERROR: {batch_path} not found", file=sys.stderr)
            sys.exit(1)
        
        with open(batch_path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        batch_stats[batch_file] = len(items)
        all_items.extend(items)
    
    # Basic counts
    total_items = len(all_items)
    unique_ids = len(set(item['id'] for item in all_items))
    
    # Ground truth distribution
    ground_truths = Counter(item.get('ground_truth') for item in all_items)
    
    # Task types
    task_types = Counter(item.get('type') for item in all_items)
    
    # Systems
    systems = Counter(item.get('system_id') for item in all_items if 'system_id' in item)
    items_with_system = sum(1 for item in all_items if 'system_id' in item)
    items_without_system = total_items - items_with_system
    
    # Dialogues
    dialogues = defaultdict(list)
    for item in all_items:
        if 'dialogue_id' in item:
            dialogues[item['dialogue_id']].append(item)
    
    dialogue_turns = []
    if dialogues:
        dialogue_turns = [len(turns) for turns in dialogues.values()]
    
    # Systems directory
    systems_dir = Path("systems")
    all_system_files = list(systems_dir.glob("*.json")) if systems_dir.exists() else []
    systems_defined = {f.stem for f in all_system_files}
    systems_used = set(systems.keys())
    systems_unused = systems_defined - systems_used
    
    stats = {
        "total_items": total_items,
        "unique_ids": unique_ids,
        "id_range": {
            "min": min(item['id'] for item in all_items),
            "max": max(item['id'] for item in all_items),
        },
        "ground_truth_distribution": dict(sorted(ground_truths.items())),
        "task_types": dict(sorted(task_types.items(), key=lambda x: -x[1])),
        "num_task_types": len(task_types),
        "systems": {
            "used_count": len(systems),
            "defined_count": len(systems_defined),
            "used_systems": sorted(systems.keys()),
            "unused_systems": sorted(systems_unused),
        },
        "items_with_system_id": items_with_system,
        "items_without_system_id": items_without_system,
        "dialogues": {
            "total_dialogues": len(dialogues),
            "turns_per_dialogue": {
                "min": min(dialogue_turns) if dialogue_turns else 0,
                "max": max(dialogue_turns) if dialogue_turns else 0,
                "avg": sum(dialogue_turns) / len(dialogue_turns) if dialogue_turns else 0,
            },
        },
        "batches": dict(sorted(batch_stats.items())),
    }
    
    return stats


def print_pretty(stats: dict):
    """Print statistics in human-readable format."""
    
    print("=" * 70)
    print("ChaosBench-Logic Dataset Statistics")
    print("=" * 70)
    
    print(f"\nTOTAL ITEMS: {stats['total_items']}")
    print(f"UNIQUE IDS: {stats['unique_ids']} (range: {stats['id_range']['min']} to {stats['id_range']['max']})")
    
    print("\nGROUND TRUTH LABELS:")
    for label, count in sorted(stats['ground_truth_distribution'].items()):
        pct = 100.0 * count / stats['total_items']
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nTASK TYPES ({stats['num_task_types']} total):")
    for task_type, count in sorted(stats['task_types'].items(), key=lambda x: -x[1]):
        print(f"  {task_type:20s}: {count:3d}")
    
    print(f"\nSYSTEMS:")
    print(f"  Used in dataset: {stats['systems']['used_count']}")
    print(f"  Defined in systems/: {stats['systems']['defined_count']}")
    print(f"  Unused systems: {', '.join(stats['systems']['unused_systems'])}")
    
    print(f"\nSYSTEM ID FIELD:")
    print(f"  Items with system_id: {stats['items_with_system_id']} ({100.0*stats['items_with_system_id']/stats['total_items']:.1f}%)")
    print(f"  Items without system_id: {stats['items_without_system_id']} ({100.0*stats['items_without_system_id']/stats['total_items']:.1f}%)")
    
    print(f"\nDIALOGUES:")
    print(f"  Total dialogues: {stats['dialogues']['total_dialogues']}")
    if stats['dialogues']['total_dialogues'] > 0:
        turns = stats['dialogues']['turns_per_dialogue']
        print(f"  Turns per dialogue: min={turns['min']}, max={turns['max']}, avg={turns['avg']:.1f}")
    
    print(f"\nBATCHES:")
    for batch, count in sorted(stats['batches'].items()):
        print(f"  {batch:40s}: {count:3d} items")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compute ChaosBench-Logic dataset statistics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    
    args = parser.parse_args()
    
    stats = compute_stats(args.data_dir)
    
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_pretty(stats)


if __name__ == "__main__":
    main()
