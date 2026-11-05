"""Validate supported real workloads before admission or first dispatch."""
import hashlib
import json
from pathlib import Path


def assert_execution_binding(spec, environ=None, *, repo=None):
    """Bind an explicit full-test client width to the shared factory environment."""
    if spec.get('workload_kind') in ('gepa-solver-template-pair', 'gepa-perception-template-pair', 'external-experience-pair', 'external-slot-knowledge-pair', 'solver-output-protocol-pair', 'intent-slot-node-ablation'):
        import os
        if spec['workload_kind'] == 'intent-slot-node-ablation':
            from intent_slot_workload import NodeWorkload as PairWorkload
        elif spec['workload_kind'] == 'solver-output-protocol-pair':
            from solver_protocol_workload import ProtocolWorkload as PairWorkload
        elif spec['workload_kind'] == 'external-slot-knowledge-pair':
            from external_slot_workload import ExternalSlotWorkload as PairWorkload
        elif spec['workload_kind'] == 'external-experience-pair':
            from external_experience_workload import ExternalExperienceWorkload as PairWorkload
        elif spec['workload_kind'] == 'gepa-perception-template-pair':
            from gepa_perception_workload import PerceptionWorkload as PairWorkload
        else:
            from gepa_pair_workload import PairWorkload
        config = PairWorkload(spec['job_config']).config
        environ = os.environ if environ is None else environ
        if (spec.get('execution') != config['execution']
                or environ.get('SAEA_ADMITTED_LEVEL') != str(config['execution']['admitted_level'])
                or environ.get('SAEA_DEPLOYMENT_PROFILE') != spec.get('profile')):
            raise ValueError('paired GEPA execution differs from registered width/profile')
        return
    if spec.get('workload_kind') == 'gepa-solver-template-probe':
        import os
        environ = os.environ if environ is None else environ
        if (spec.get('execution') != {'admitted_level': 96}
                or environ.get('SAEA_ADMITTED_LEVEL') != '96'
                or environ.get('SAEA_DEPLOYMENT_PROFILE') != spec.get('profile')):
            raise ValueError('GEPA probe requires registered C96 environment')
        return
    if spec.get('workload_kind') == 'tfrl-h1-fresh-observation' and spec.get('comparison') is not None:
        import os
        from tfrl_h1_observation import comparison_config
        cfg = comparison_config(spec['comparison'])
        environ = os.environ if environ is None else environ
        if (environ.get('SAEA_ADMITTED_LEVEL') != str(cfg['admitted_level'])
                or environ.get('SAEA_DEPLOYMENT_PROFILE') != spec.get('profile')):
            raise ValueError('H1 thinking comparison requires registered C96 environment')
        return
    if spec.get('workload_kind') != 'final-report-method':
        return
    import os
    environ = os.environ if environ is None else environ
    if repo is None:
        raise ValueError('full-test execution requires registered deployment configuration')
    from final_report_workload import execution_binding
    expected = execution_binding(repo, spec['job_config'])
    if spec.get('profile') != spec['job_config']['runtime']['profile']:
        raise ValueError('full-test profile differs from registered configuration')
    if (spec.get('execution') != expected
            or spec.get('job_config', {}).get('execution') != {'admitted_level': 96}
            or environ.get('SAEA_ADMITTED_LEVEL') != '96'
            or environ.get('SAEA_DEPLOYMENT_PROFILE') != spec.get('profile')):
        raise ValueError('full-test execution requires registered C96 per instance environment')


def validate_workload(repo, data, name, spec):
    if name != spec["cell"] or name != spec["experiment"]:
        raise ValueError("registered workload identity mismatch")
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
    exp = json.loads(resolve_asset(repo, "configs/experiment/" + name + ".json").read_bytes())
    cell, = exp["cells"]
    if cell["run_id"] != spec["run_id"] or cell["attempt_id"] != spec["attempt_id"]:
        raise ValueError("registered attempt differs")
    view = json.loads(resolve_asset(repo, "configs/dataset/" + cell["dataset_fragment"] + ".json").read_bytes())
    if spec.get("workload_kind") in ("concurrency-sweep-real100", "concurrency-sweep-real500"):
        count = 500 if spec["workload_kind"].endswith("real500") else 100
        import importlib.util
        module_path = Path(__file__).with_name("concurrency_sweep_workload.py")
        module_spec = importlib.util.spec_from_file_location("saea_concurrency_sweep_workload", module_path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError("cannot load concurrency workload validator")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        validate_manifest = module.validate_manifest
        manifest_path = Path(spec["manifest_path"])
        roster_path = Path(spec["source_roster_path"])
        checked = validate_manifest(
            manifest_path,
            source_roster_path=roster_path,
            expected_profile_id=spec["profile"],
            expected_profile_sha256=spec["profile_sha256"],
            expected_count=count,
        )
        if checked["manifest_sha256"] != spec["manifest_sha256"]:
            raise ValueError("concurrency manifest hash differs")
        if spec.get("rows") != count or spec.get("new_model_calls") != 0:
            raise ValueError(f"concurrency workload registration is not the fixed {count}-row held plan")
        # The runner consumes the experiment's dataset view, not the manifest.
        # Bind the exact ordered rows before either admission or dispatch.
        if view.get("count") != count or view.get("items") != checked["items"]:
            raise ValueError("concurrency execution view differs from fixed manifest")
        identity = hashlib.sha256("\n".join(
            sorted("slurp/" + sid for sid in checked["sample_ids"])
        ).encode()).hexdigest()
        if view.get("split_identity_hash") != identity or spec.get("split_identity_hash") != identity:
            raise ValueError("concurrency split identity differs")
        unsigned = {key: value for key, value in view.items() if key != "runtime_view_sha256"}
        view_digest = hashlib.sha256(json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if view.get("runtime_view_sha256") != view_digest:
            raise ValueError("concurrency runtime view hash differs")
        if count == 500:
            if view.get("source_binding_sha256") != checked["source_roster_content_sha256"]:
                raise ValueError("concurrency full500 source binding differs")
            if spec.get("runtime_view_sha256") != view_digest:
                raise ValueError("concurrency full500 registered runtime view differs")
            if spec.get("source_roster_sha256") != checked["source_roster_sha256"]:
                raise ValueError("concurrency full500 registered roster differs")
        return count
    if spec.get("workload_kind") == "tfgrpo-dense-h2-paired":
        from tfgrpo_dense_h2_job import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "tfgrpo-render-h2-three-arm":
        from tfgrpo_render_h2_job import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") in ("tfgrpo-full-h2-paired", "learning-full-h2-paired"):
        if spec["workload_kind"] == "learning-full-h2-paired" and "evaluation_config" not in spec:
            raise ValueError("strategy-neutral H2 requires explicit evaluation configuration")
        from tfgrpo_h2_job import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "final-report-method":
        from final_report_workload import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "intent-slot-node-ablation":
        from intent_slot_workload import validate
        return validate(repo,data,name,spec,exp,view)
    if spec.get("workload_kind") == "solver-output-protocol-pair":
        from solver_protocol_workload import validate
        return validate(repo,data,name,spec,exp,view)
    if spec.get("workload_kind") == "external-slot-knowledge-pair":
        from external_slot_workload import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "external-experience-pair":
        from external_experience_workload import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "gepa-perception-template-pair":
        from gepa_perception_workload import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "gepa-solver-template-pair":
        from gepa_pair_workload import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "gepa-solver-template-probe":
        from gepa_probe_workload import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "tfrl-h1-fresh-observation":
        from tfrl_h1_observation import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "tfrl-hardset-a0-observation":
        from tfrl_hardset_a0_registration import validate
        return validate(repo, data, name, spec, exp, view)
    if spec.get("workload_kind") == "a0-fold-a-c2":
        from fold_a_c2_registration import validate
        return validate(repo,data,name,spec,exp,view)
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
