#!/usr/bin/env python3
"""Validate disjoint deployment/state configuration without touching services."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(document, repo):
    require(set(document) == {"schema", "card_count", "lanes"}
            and document["schema"] == "saea-resource-partition-v1", "partition schema differs")
    count = document["card_count"]
    require(type(count) is int and count > 0, "positive physical card count required")
    require(isinstance(document["lanes"], list) and document["lanes"], "nonempty lanes required")
    names, cards, ports, roots = set(), set(), set(), []
    result = []
    repo = Path(repo).resolve()
    for lane in document["lanes"]:
        require(set(lane) == {"name", "profile", "state_root"}, "exact lane fields required")
        name, profile_id = lane["name"], lane["profile"]
        require(isinstance(name, str) and name and name not in names, "unique lane name required")
        require(isinstance(profile_id, str) and profile_id and Path(profile_id).name == profile_id
                and profile_id not in (".", ".."), "confined deployment profile required")
        names.add(name)
        path = (repo / "configs/deployment" / (profile_id + ".json")).resolve()
        require(path.is_relative_to(repo), "profile escapes repository")
        profile = json.loads(path.read_bytes())
        model_id = profile["model_fragment"]
        require(isinstance(model_id, str) and Path(model_id).name == model_id
                and model_id not in (".", ".."), "confined model required")
        model_path = (repo / "configs/model" / (model_id + ".json")).resolve()
        require(model_path.is_relative_to(repo), "model escapes repository")
        model = json.loads(model_path.read_bytes())
        raw_root = lane["state_root"]
        require(isinstance(raw_root, str) and Path(raw_root).is_absolute()
                and ".." not in Path(raw_root).parts, "absolute canonical state root required")
        root = Path(raw_root).resolve()
        require(str(root) == raw_root and len(root.parts) >= 4, "scoped canonical state root required")
        require(all(not root.is_relative_to(old) and not old.is_relative_to(root) for old in roots),
                "lane state roots overlap")
        roots.append(root)
        instances = profile["instances"]
        require(isinstance(instances, list) and instances, "nonempty deployment instances required")
        lane_cards, lane_ports = [], []
        for instance in instances:
            selected, port = instance["cards"], instance["port"]
            require(isinstance(selected, list) and len(selected) == profile["tensor_parallel_size"]
                    and all(type(c) is int and 0 <= c < count for c in selected), "invalid physical cards")
            require(len(set(selected)) == len(selected) and not cards.intersection(selected), "physical cards overlap")
            require(type(port) is int and 0 < port < 65536 and port not in ports, "invalid or overlapping port")
            cards.update(selected); ports.add(port); lane_cards.extend(selected); lane_ports.append(port)
        endpoint = urlparse(model["base_url"])
        require(endpoint.port == instances[0]["port"], "model endpoint differs from first deployment instance")
        result.append({"name": name, "profile": profile_id, "cards": lane_cards,
                       "ports": lane_ports, "state_root": str(root),
                       "guard": str(root / "locks/generation.guard")})
    return {"schema": "saea-resource-partition-validated-v1", "lanes": result,
            "used_cards": sorted(cards), "new_model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.config.read_bytes()), args.repo), sort_keys=True))
