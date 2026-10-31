"""Validate supported real workloads before admission or first dispatch."""
import json


def validate_workload(repo, data, name, spec):
    if name != spec["cell"] or name != spec["experiment"]:
        raise ValueError("registered workload identity mismatch")
    exp = json.loads((repo / "configs/experiment" / (name + ".json")).read_bytes())
    cell, = exp["cells"]
    if cell["run_id"] != spec["run_id"] or cell["attempt_id"] != spec["attempt_id"]:
        raise ValueError("registered attempt differs")
    view = json.loads((repo / "configs/dataset" / (cell["dataset_fragment"] + ".json")).read_bytes())
    if spec.get("workload_kind") == "tfrl-real-seed-half":
        from tfrl_current_generation import validate_half
        return validate_half(repo, name, spec, exp, view)
    if spec.get("workload_kind") == "tfrl-real-banked-half":
        from tfrl_current_generation import validate_half
        iteration=spec.get("iteration")
        if type(iteration) is not int or iteration not in (2,3):
            raise ValueError("banked workload requires iteration2/3")
        return validate_half(repo,name,spec,exp,view,data=data,iteration=iteration)
    if spec.get("workload_kind") == "tfrl-context-repair-complement":
        from tfrl_h1_remaining import validate
        return validate(repo,data,name,spec,exp,view)
    ids = [row["sample_id"] for row in view["items"]]
    if len(ids) != spec["rows"] or view["count"] != spec["rows"] or len(set(ids)) != len(ids) or view["split_identity_hash"] != spec["split_identity_hash"]:
        raise ValueError("registered population differs")
    if spec.get("workload_kind") == "a0-exact-remaining":
        from score_a0_full1080 import pin, validate_recipe, verify_sources
        if name != "slurp-a0-remaining353-c2-v1" or spec["rows"] != 353 or spec["run_id"] != "SAEA-SLURP-A0-REMAINING353-C2-V1":
            raise ValueError("unexpected A0 continuation")
        recipe = json.loads(pin(repo, spec["source_read_recipe"]).read_bytes()); validate_recipe(recipe)
        sources = verify_sources(repo, data, recipe)
        if view != sources["remaining_view"] or spec["source_custody"] != recipe["custody_summary"]:
            raise ValueError("continuation differs from exact source-bound missing population")
        if spec.get("profile_selection_state") != "selected":
            raise ValueError("measured engineering profile selection is pending")
        selection = json.loads(pin(data, spec["profile_selection"]).read_bytes())
        from bind_a0_selected_profile import verify_selection
        verify_selection(data, selection)
        if selection["profile"] != spec["profile"] or selection.get("complete_roster_comparison") is not True:
            raise ValueError("measured profile selection differs")
    elif spec["rows"] == 500 and name in {
        "slurp-real500-tp4-s96-v1", "slurp-real500-tp4-s128-v1", "slurp-real500-tp8-s256-v1", "slurp-real500-tp8-s384-v1"
    }:
        from prepare_real_engine_roster import prepare
        _, expected = prepare(repo / "configs/dataset/slurp-speech-native-evo-fold-b-1080.json")
        if view != expected:
            raise ValueError("real500 roster differs from fixed selection")
    else:
        raise ValueError("unsupported registered real workload")
    return spec["rows"]
