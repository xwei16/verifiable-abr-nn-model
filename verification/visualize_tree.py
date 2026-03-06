#!/usr/bin/env python3
"""
Visualize bound splitting tree from JSONL logs.

Usage:
    python visualize_tree.py <jsonl_log_file> [--output <output_prefix>]

Tree structure produced by lirpa_pensieve:

  [Round-0 Root]  (parent_id=None)
      ├── [Round-0 Safe leaf]  (parent_id=Round-0 Root)
      │       └── [Round-1 Root]  (parent_id=Round-0 Safe leaf)
      │               ├── [Round-1 Safe leaf]
      │               └── ...
      └── ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    nodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))
    return nodes


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class TreeNode:
    def __init__(self, d: Dict):
        self.node_id           = d["node_id"]
        self.parent_id         = d.get("parent_id")
        self.depth             = d.get("depth", 0)
        self.level             = d.get("level", 0)
        self.throughput_lb     = d.get("throughput_lb", 0.0)
        self.throughput_ub     = d.get("throughput_ub", 0.0)
        self.throughput_all_lb = d.get("throughput_all_lb", [])
        self.throughput_all_ub = d.get("throughput_all_ub", [])
        self.download_time_lb  = d.get("download_time_lb", [])
        self.download_time_ub  = d.get("download_time_ub", [])
        self.last_br_lb        = d.get("last_br_lb", 0.0)
        self.last_br_ub        = d.get("last_br_ub", 0.0)
        self.current_br_lb     = d.get("current_br_lb", 0.0)
        self.current_br_ub     = d.get("current_br_ub", 0.0)
        self.buffer_lb         = d.get("buffer_lb", 0.0)
        self.buffer_ub         = d.get("buffer_ub", 0.0)
        self.status            = d.get("status")
        self.action            = d.get("action")
        self.children: List["TreeNode"] = []

    def add_child(self, child: "TreeNode"):
        self.children.append(child)


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class BoundSplittingTree:
    """
    Builds and visualises the full verification tree.

    Node relationships after the lirpa_pensieve fix:
      - Round-N roots have parent_id = the safe-leaf node_id from round N-1
        that spawned them (None for the very first root).
      - Safe leaves have parent_id = their round's root.

    So the tree is a single connected structure across all rounds.
    """

    def __init__(self, node_dicts: List[Dict]):
        self.nodes_by_id: Dict[int, TreeNode] = {}
        self.roots: List[TreeNode] = []   # nodes whose parent_id is None

        for d in node_dicts:
            node = TreeNode(d)
            self.nodes_by_id[node.node_id] = node

        for node in self.nodes_by_id.values():
            if node.parent_id is None:
                self.roots.append(node)
            else:
                parent = self.nodes_by_id.get(node.parent_id)
                if parent:
                    parent.add_child(node)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        stats = {
            "total_nodes":  len(self.nodes_by_id),
            "safe_regions": 0,
            "split_nodes":  0,
            "safe_actions": {},
        }
        for node in self.nodes_by_id.values():
            if node.status == "SAFE":
                stats["safe_regions"] += 1
                stats["safe_actions"].setdefault(node.action, 0)
                stats["safe_actions"][node.action] += 1
            elif node.status == "SPLIT":
                stats["split_nodes"] += 1
        return stats

    # ------------------------------------------------------------------
    # ASCII output
    # ------------------------------------------------------------------

    def _node_header(self, node: TreeNode) -> str:
        icon = "🟡" if node.status == "SPLIT" else "🟢"
        tag  = f"[Node {node.node_id}]  Round {node.level}  Depth {node.depth}"
        tput = f"Throughput: [{node.throughput_lb:.4f}, {node.throughput_ub:.4f}]"
        if node.status == "SAFE":
            return f"{icon} {tag}\n   {tput}  ✓ Action {node.action}"
        else:
            return f"{icon} {tag} — Split root\n   {tput}"

    def _ascii_subtree(self, node: TreeNode, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        lines = []
        for raw_line in self._node_header(node).splitlines():
            lines.append(prefix + connector + raw_line)
            connector = "    "   # subsequent lines of the same node indent cleanly

        for i, child in enumerate(node.children):
            last_child = (i == len(node.children) - 1)
            lines.append(self._ascii_subtree(child, child_prefix, last_child))

        return "\n".join(lines)

    def print_tree(self) -> str:
        parts = []
        for root in sorted(self.roots, key=lambda n: n.node_id):
            parts.append(self._ascii_subtree(root, prefix="", is_last=True))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # DOT / SVG output
    # ------------------------------------------------------------------

    def to_dot_format(self) -> str:
        """
        Emit one subgraph cluster per (level, status) combination so that
        rounds are visually grouped.  Edges follow parent→child.
        """
        dot_lines = [
            "digraph BoundSplittingTree {",
            "  rankdir=TB;",
            "  node [shape=box, style=\"rounded,filled\", fontsize=10];",
            "  compound=true;",
            "  splines=ortho;",
        ]

        # Group by level
        by_level: Dict[int, List[TreeNode]] = {}
        for node in self.nodes_by_id.values():
            by_level.setdefault(node.level, []).append(node)

        for level in sorted(by_level.keys()):
            dot_lines.append(f'  subgraph cluster_level_{level} {{')
            dot_lines.append(f'    label="Round {level}";')
            dot_lines.append(f'    style=dotted;')

            for node in sorted(by_level[level], key=lambda n: n.node_id):
                # Build label
                tput = f"Throughput: [{node.throughput_lb:.4f}, {node.throughput_ub:.4f}]"

                all_tput = ""
                if node.throughput_all_lb and node.throughput_all_ub:
                    all_tput = "\\nAll Throughput:\\n" + "".join(
                        f"  Slot {i}: [{node.throughput_all_lb[i]:.3f},"
                        f" {node.throughput_all_ub[i]:.3f}]\\n"
                        for i in range(8)
                    )

                dt_str = "\\nDownload Time:\\n"
                if node.download_time_lb and node.download_time_ub:
                    dt_str += "".join(
                        f"  Slot {i}: [{node.download_time_lb[i]:.3f},"
                        f" {node.download_time_ub[i]:.3f}]\\n"
                        for i in range(8)
                    )

                meta = (
                    f"\\nLast BR: [{node.last_br_lb:.4f}, {node.last_br_ub:.4f}]"
                    f"\\nBuffer:  [{node.buffer_lb:.4f}, {node.buffer_ub:.4f}]"
                    f"\\nNode {node.node_id} | Round {node.level} | Depth {node.depth}"
                )

                label = tput + all_tput + dt_str + meta
                if node.status == "SAFE":
                    label += f"\\n✓ Action: {node.action}"

                color = "lightgreen" if node.status == "SAFE" else "lightyellow"
                dot_lines.append(
                    f'    node_{node.node_id} [label="{label}", fillcolor={color}];'
                )

            dot_lines.append("  }")

        # Edges
        for node in self.nodes_by_id.values():
            for child in node.children:
                dot_lines.append(f"  node_{node.node_id} -> node_{child.node_id};")

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def save_dot_file(self, path: Path):
        path.write_text(self.to_dot_format())
        print(f"Saved DOT file to {path}")

    def generate_svg(self, path: Path):
        try:
            import graphviz
        except ImportError:
            print("Warning: graphviz not installed.  pip install graphviz")
            return
        stem = str(path).replace(".dot", "").replace(".svg", "")
        try:
            graphviz.Source(self.to_dot_format()).render(stem, format="svg", cleanup=True)
            print(f"Saved SVG to {stem}.svg")
        except Exception as e:
            print(f"Warning: could not generate SVG: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize bound splitting tree from JSONL logs"
    )
    parser.add_argument("jsonl_log", type=str, help="Path to .jsonl log file")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-ascii", action="store_true")
    parser.add_argument("--dot-only", action="store_true")
    args = parser.parse_args()

    jsonl_file = Path(args.jsonl_log)
    if not jsonl_file.exists():
        print(f"Error: {jsonl_file} not found")
        sys.exit(1)

    node_dicts = load_jsonl(jsonl_file)
    tree = BoundSplittingTree(node_dicts)

    print("=" * 70)
    print("BOUND SPLITTING TREE ANALYSIS")
    print("=" * 70)
    stats = tree.get_statistics()
    print(f"Total Nodes  : {stats['total_nodes']}")
    print(f"Split Nodes  : {stats['split_nodes']}")
    print(f"Safe Regions : {stats['safe_regions']}")
    if stats["safe_actions"]:
        print("\nSafe Actions:")
        for action, count in sorted(stats["safe_actions"].items()):
            print(f"  Action {action}: {count} region(s)")
    print("=" * 70)
    print()

    if not args.no_ascii:
        print("TREE VISUALIZATION:")
        print("-" * 70)
        print(tree.print_tree())
        print("-" * 70)
        print()

    base = Path(args.output) if args.output else jsonl_file.parent / jsonl_file.stem
    dot_path = Path(str(base) + ".dot")
    tree.save_dot_file(dot_path)

    if not args.dot_only:
        tree.generate_svg(Path(str(base) + ".svg"))

    print(f"\nFiles written to {jsonl_file.parent}/")


if __name__ == "__main__":
    main()