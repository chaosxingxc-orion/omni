"""Learning-only, four-state diagnostics; never changes runtime or gold."""
import argparse
import collections
import json
from pathlib import Path

PAIRS = {
    "calendar_set": [("person", "relation"), ("time", "timeofday"), ("date", "time")],
    "play_music": [("artist_name", "song_name"), ("artist_name", "music_genre"), ("song_name", "music_genre")],
    "cooking_recipe": [("food_type", "ingredient")],
    "weather_query": [("time", "timeofday"), ("date", "time")],
}


def state(types, a, b):
    return "both" if a in types and b in types else "A_only" if a in types else "B_only" if b in types else "neither"


def analyze(rows):
    learning = [r for r in rows if r["split"] == "learning"]
    assert len(learning) == 476
    assert len({r["sample_id"] for r in learning}) == 476
    assert all(r["candidate_id"] == r["gold_intent"] for r in learning)
    result = {}
    for intent, pairs in PAIRS.items():
        group = [r for r in learning if r["gold_intent"] == intent]
        for a, b in pairs:
            matrix = {s: collections.Counter() for s in ["A_only", "B_only", "both", "neither"]}
            for r in group:
                gold = state(set(r["gold_types"]), a, b)
                predicted = state(set(r["slot_types"]), a, b) if r["valid"] else "invalid"
                matrix[gold][predicted] += 1
            assert sum(sum(v.values()) for v in matrix.values()) == len(group)
            result[f"{intent}:{a}/{b}"] = {
                "A": a, "B": b, "samples": len(group), "matrix": matrix,
                "opposite_single_type_predictions": matrix["A_only"]["B_only"] + matrix["B_only"]["A_only"],
                "single_expanded_to_both": matrix["A_only"]["both"] + matrix["B_only"]["both"],
                "coexistence_reduced_to_single": matrix["both"]["A_only"] + matrix["both"]["B_only"],
                "neither_expanded": sum(v for k, v in matrix["neither"].items() if k not in ["neither", "invalid"]),
            }
    return {"scope": "learning476_gold_intent_only", "causal_attribution": False,
            "note": "Four-state projections are diagnostic signals, not proof of semantic substitution; pairs overlap and must not be summed as unique errors.",
            "pairs": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze([json.loads(s) for s in args.scores.read_text().splitlines() if s.strip()])
    result["source"] = str(args.scores)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
