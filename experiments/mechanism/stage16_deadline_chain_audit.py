"""Audit the combinatorics and scheduling limits of multi-item revocation.

The experiment separates three claims that are easy to conflate:
1. irreversible deadlines induce a chain of required policy masks;
2. a chain reduces the number of logical counterfactuals, not their total work;
3. lagging execution shifts work and requires a replay buffer; it does not make
   state transitions disappear.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import heapq
import itertools
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/stage16_deadline_chain_audit.json")
    return parser.parse_args()


def masks_at_deadlines(items):
    deadlines = sorted({item["deadline"] for item in items})
    masks = [tuple()]
    for deadline in deadlines:
        masks.append(tuple(sorted(item["id"] for item in items if item["deadline"] <= deadline)))
    return masks


def prospective_jobs(items):
    """One exact trajectory for each distinct nonempty deadline mask."""
    by_id = {item["id"]: item for item in items}
    jobs = []
    for index, mask in enumerate(masks_at_deadlines(items)[1:], start=1):
        deadline = max(by_id[item_id]["deadline"] for item_id in mask)
        start = min(by_id[item_id]["arrival"] for item_id in mask)
        jobs.append({
            "branch": index,
            "mask": mask,
            "start": start,
            "deadline": deadline,
            "transitions": deadline - start + 1,
        })
    return jobs


def simulate_edf(items, capacity):
    """Run FIFO-within-branch, EDF-across-branch unit transition scheduling."""
    jobs = prospective_jobs(items)
    last_deadline = max(job["deadline"] for job in jobs)
    queues = {job["branch"]: [] for job in jobs}
    job_by_branch = {job["branch"]: job for job in jobs}
    processed = defaultdict(int)
    generated = defaultdict(int)
    missed = []
    peak_backlog = 0
    peak_retained_input_tokens = 0
    work_by_time = []
    for now in range(last_deadline + 1):
        for job in jobs:
            if job["start"] <= now <= job["deadline"]:
                queues[job["branch"]].append(now)
                generated[job["branch"]] += 1
        used = 0
        ready = []
        for branch, queue in queues.items():
            if queue:
                job = job_by_branch[branch]
                heapq.heappush(ready, (job["deadline"], branch))
        while ready and used < capacity:
            _, branch = heapq.heappop(ready)
            queues[branch].pop(0)
            processed[branch] += 1
            used += 1
            if queues[branch]:
                job = job_by_branch[branch]
                heapq.heappush(ready, (job["deadline"], branch))
        backlog = sum(len(queue) for queue in queues.values())
        peak_backlog = max(peak_backlog, backlog)
        pending_times = [token_time for queue in queues.values() for token_time in queue]
        retained = now - min(pending_times) + 1 if pending_times else 0
        peak_retained_input_tokens = max(peak_retained_input_tokens, retained)
        for job in jobs:
            if now == job["deadline"] and queues[job["branch"]]:
                missed.append({
                    "branch": job["branch"],
                    "deadline": now,
                    "unfinished_transitions": len(queues[job["branch"]]),
                })
        work_by_time.append(used)
    total_required = sum(job["transitions"] for job in jobs)
    return {
        "capacity_transitions_per_live_step": capacity,
        "feasible": not missed and sum(processed.values()) == total_required,
        "missed": missed,
        "total_required_transitions": total_required,
        "total_processed_transitions": sum(processed.values()),
        "peak_scheduled_transitions_per_step": max(work_by_time),
        "mean_scheduled_transitions_per_step": sum(work_by_time) / len(work_by_time),
        "peak_transition_backlog": peak_backlog,
        "peak_retained_input_tokens": peak_retained_input_tokens,
    }


def scenario(name, items):
    chain_masks = masks_at_deadlines(items)
    jobs = prospective_jobs(items)
    k = len(items)
    eager_work = sum(job["transitions"] for job in jobs)
    active_counts = []
    for now in range(max(job["deadline"] for job in jobs) + 1):
        active_counts.append(sum(job["start"] <= now <= job["deadline"] for job in jobs))
    schedules = [simulate_edf(items, capacity) for capacity in range(1, len(jobs) + 2)]
    minimum_feasible = next(
        (row["capacity_transitions_per_live_step"] for row in schedules if row["feasible"]),
        None,
    )
    return {
        "name": name,
        "items": items,
        "all_subset_masks": 2**k,
        "deadline_chain_masks_including_empty": len(chain_masks),
        "chain_masks": chain_masks,
        "logical_branch_reduction_factor_vs_boolean_lattice": (2**k) / len(chain_masks),
        "prospective_jobs": jobs,
        "eager_total_transitions": eager_work,
        "eager_peak_transitions_per_live_step": max(active_counts),
        "minimum_integer_capacity_for_feasibility": minimum_feasible,
        "edf_schedules": schedules,
    }


def exhaustive_chain_check(max_items=7):
    checked = 0
    violations = []
    for k in range(1, max_items + 1):
        for deadlines in itertools.product(range(1, 4), repeat=k):
            items = [
                {"id": index, "arrival": 0, "deadline": deadline}
                for index, deadline in enumerate(deadlines)
            ]
            masks = masks_at_deadlines(items)
            checked += 1
            if len(masks) > len(set(deadlines)) + 1 or any(
                not set(left).issubset(right)
                for left, right in zip(masks, masks[1:])
            ):
                violations.append({"deadlines": deadlines, "masks": masks})
    return {"assignments_checked": checked, "violations": violations}


def main():
    args = parse_args()
    scenarios = [
        scenario("staggered_three", [
            {"id": 0, "arrival": 3, "deadline": 18},
            {"id": 1, "arrival": 7, "deadline": 30},
            {"id": 2, "arrival": 12, "deadline": 45},
        ]),
        scenario("bucketed_six", [
            {"id": 0, "arrival": 2, "deadline": 24},
            {"id": 1, "arrival": 5, "deadline": 24},
            {"id": 2, "arrival": 9, "deadline": 40},
            {"id": 3, "arrival": 13, "deadline": 40},
            {"id": 4, "arrival": 16, "deadline": 56},
            {"id": 5, "arrival": 20, "deadline": 56},
        ]),
    ]
    exhaustive = exhaustive_chain_check()
    output = {
        "status": "stage_16_deadline_chain_and_scheduler_audit",
        "claims": {
            "supported": [
                "irreversible deadline masks are nested and number at most b+1 for b distinct deadline buckets",
                "EDF can move counterfactual transitions into slack when sufficient capacity exists",
            ],
            "not_supported": [
                "lagging shadows reduce the number of required exact state transitions",
                "lagging shadows are free in memory; pending inputs must remain replayable",
                "deadline chains alone reduce b prospective trajectories to one",
            ],
        },
        "exhaustive_chain_check": exhaustive,
        "scenarios": scenarios,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, output, newline=True)
    print(json.dumps({
        "output": str(destination),
        "chain_assignments_checked": exhaustive["assignments_checked"],
        "chain_violations": len(exhaustive["violations"]),
        "scenarios": [
            {
                "name": row["name"],
                "all_subsets": row["all_subset_masks"],
                "chain_masks": row["deadline_chain_masks_including_empty"],
                "minimum_capacity": row["minimum_integer_capacity_for_feasibility"],
                "eager_work": row["eager_total_transitions"],
            }
            for row in scenarios
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
