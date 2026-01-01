#!/usr/bin/env python3
"""
Aggregate results from multiple model evaluations into markdown tables.

Usage:
    python scripts/aggregate_results.py [output_file]
    
This script reads from results/{model}_{mode}/summary.json and generates
a formatted markdown table suitable for README.md or RESULTS.md.

If output_file is specified, writes to that file. Otherwise prints to stdout.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def discover_results(results_dir: str = "results") -> dict:
    """
    Discover all available result directories and load summaries.
    
    Returns:
        Dict mapping (model, mode) -> summary data
    """
    
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"WARNING: {results_path} not found", file=sys.stderr)
        return results
    
    # Find all {model}_{mode} directories
    for result_dir in sorted(results_path.iterdir()):
        if not result_dir.is_dir():
            continue
        
        # Parse directory name
        parts = result_dir.name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        
        model, mode = parts
        
        # Load summary.json
        summary_file = result_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                results[(model, mode)] = summary
    
    return results


def format_markdown_table(results: dict) -> str:
    """
    Format results as a markdown table.
    
    Args:
        results: Dict from discover_results()
    
    Returns:
        Formatted markdown table string
    """
    
    if not results:
        return "No results found."
    
    lines = []
    lines.append("### Performance Summary")
    lines.append("")
    
    # Sort by (model, mode)
    sorted_results = sorted(results.items())
    
    # Header
    lines.append("| Model | Mode | Overall Accuracy | Dialogue Accuracy | Task Types |")
    lines.append("|-------|------|------------------|-------------------|------------|")
    
    # Rows
    for (model, mode), summary in sorted_results:
        overall_acc = summary.get('overall_accuracy')
        dialogue_acc = summary.get('dialogue_accuracy')
        task_acc = summary.get('task_accuracy', {})
        num_task_types = len(task_acc) if task_acc else 0
        
        # Format percentages
        overall_str = f"{overall_acc*100:.1f}%" if overall_acc is not None else "N/A"
        dialogue_str = f"{dialogue_acc*100:.1f}%" if dialogue_acc is not None else "N/A"
        
        lines.append(f"| {model} | {mode} | {overall_str} | {dialogue_str} | {num_task_types} |")
    
    lines.append("")
    
    # Per-task accuracy table
    lines.append("### Accuracy by Task Type")
    lines.append("")
    
    # Collect all task types across all results
    all_task_types = set()
    for summary in results.values():
        all_task_types.update(summary.get('task_accuracy', {}).keys())
    
    all_task_types = sorted(all_task_types)
    
    if all_task_types:
        # Header
        header = "| Task Type |"
        for model, mode in sorted_results:
            header += f" {model}_{mode} |"
        lines.append(header)
        
        # Separator
        sep = "|---|"
        for _ in sorted_results:
            sep += "---|"
        lines.append(sep)
        
        # Rows
        for task_type in all_task_types:
            row = f"| {task_type} |"
            for (model, mode), summary in sorted_results:
                task_acc = summary.get('task_accuracy', {}).get(task_type)
                if task_acc is not None:
                    row += f" {task_acc*100:.1f}% |"
                else:
                    row += " — |"
            lines.append(row)
    
    return "\n".join(lines)


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Discovering results...", file=sys.stderr)
    results = discover_results()
    
    if not results:
        print("No results found in results/ directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(results)} result set(s)", file=sys.stderr)
    
    markdown = format_markdown_table(results)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(markdown)
        print(f"Written to {output_file}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
