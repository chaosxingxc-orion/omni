"""Candidate Slot stage composed solely from the existing metered native role.

Per-candidate visits are sequential for deterministic Flight custody. Concurrency
across rows remains the shared runtime's responsibility; no private model client.
"""
from __future__ import annotations

import copy
import inspect
import json

from openjiuwen.core.workflow import WorkflowComponent

from ...rails.intent_slot_pipeline import candidates, digest, parse_config, slot_results


class CandidateSlotStage(WorkflowComponent):
    def __init__(self, *, config_json, node, builder, checkpoint, actions):
        super().__init__()
        self.config = parse_config(json.loads(config_json))
        self.node, self.builder, self.checkpoint, self.actions = node, builder, checkpoint, actions

    async def invoke(self, inputs, session, context):
        from .native_episode import _ApplyRole, _state
        state = copy.deepcopy(_state(inputs))
        rows = candidates(state["perception"], self.config["max_candidates"], duplicate_policy=self.config.get("duplicate_candidate_policy", "strict"))
        raw_rows = copy.deepcopy(rows)
        if self.config.get("reference_normalizer"):
            clean = self.node.clean_source("perception", state["perception"], [])
            rows = candidates(clean, self.config["max_candidates"], duplicate_policy=self.config.get("duplicate_candidate_policy", "strict"))
        source = copy.deepcopy(state["latest_perception"]["execution_receipt"]["final_contact"])
        normalized_perception_sha256 = None
        if self.config.get("node_rails") == "sdk-handoff-v1":
            import hashlib
            from ...rails.candidate_handoff import normalized_response
            normalized_perception_sha256 = hashlib.sha256(normalized_response("perception", clean if self.config.get("reference_normalizer") else state["perception"], {}, self.config).encode()).hexdigest()
        batches = [rows] if self.config["slot_mode"] == "batched" else [[row] for row in rows]
        results, contacts = {}, []
        recovery = self.config.get("coverage_recovery")
        recovery_contacts = []
        pending_batches = list(batches)
        while pending_batches:
            batch = pending_batches.pop(0)
            if recovery:
                state["slot_recovery_pending"] = {"strategy": recovery["strategy"],
                    "original_candidate_ids": [row["candidate_id"] for row in rows],
                    "valid_results": copy.deepcopy(results), "sources": copy.deepcopy(recovery_contacts),
                    "remaining_candidate_ids": [row["candidate_id"] for row in batch] +
                        [row["candidate_id"] for pending in pending_batches for row in pending],
                    "max_extra_contacts": recovery["max_extra_contacts"]}
            state["intent_slot_context"] = {
                "candidates": batch, "perception_source": source,
                "candidate_set_sha256": digest(rows),
                **({"coverage_repair": {"reason": "missing_or_invalid_candidate_result",
                    "candidate_id": batch[0]["candidate_id"]}} if recovery and recovery_contacts else {}),
                **({"normalized_perception_sha256": normalized_perception_sha256} if normalized_perception_sha256 else {}),
            }
            self.checkpoint.capture(state=state, role_id=self.node.spec.role_id, stage="slot")
            query = self.builder(self.node.spec.role_id, copy.deepcopy(state))
            if inspect.isawaitable(query):
                query = await query
            role = await self.node.invoke_direct(query)
            applied = await _ApplyRole("slot", self.actions).invoke(
                {"state": state, "role": role}, session, context)
            state = applied["state"]
            ids = [row["candidate_id"] for row in batch]
            parsed_text = role["actor_output"]
            if recovery:
                report = self.node.coverage_result()
                parsed = report["valid"]
                if set(parsed) & set(results):
                    raise ValueError("Slot repair attempted to overwrite a valid candidate")
                results.update(parsed)
                recovery_contacts.append({"candidate_ids": ids,
                    "source": copy.deepcopy(role["execution_receipt"]["final_contact"]),
                    "valid_ids": list(parsed), "result_sha256": digest(parsed)})
                if len(recovery_contacts) == 1:
                    missing = report["missing"]
                    if len(missing) > recovery["max_extra_contacts"]:
                        raise ValueError("Slot coverage recovery budget exhausted")
                    pending_batches.extend([[row] for row in rows if row["candidate_id"] in missing])
                elif report["missing"]:
                    if not getattr(self.node, "coverage_successor", lambda: False)():
                        raise ValueError("Slot candidate recovery did not produce a valid result")
                    pending_batches.insert(0, batch)
                continue
            if self.config.get("reference_normalizer"):
                parsed_text = self.node.clean_source("slot", parsed_text, ids)
            if self.config.get("node_rails") == "sdk-handoff-v1":
                from ...rails.candidate_handoff import normalized_response
                parsed_text = normalized_response("slot", parsed_text, {"candidates": batch}, self.config)
            parsed = slot_results(parsed_text, ids)
            results.update(parsed)
            contacts.append({"candidate_ids": ids,
                             "source": copy.deepcopy(role["execution_receipt"]["final_contact"]),
                             "result_sha256": digest(parsed)})
        if recovery:
            contacts = recovery_contacts
            state.pop("slot_recovery_pending", None)
        state["intent_slot_context"] = {
            "candidates": rows, "perception_source": source,
            "candidate_set_sha256": digest(rows), "slot_results": results,
            "slot_sources": contacts,
        }
        state["intent_slot_pipeline"] = {
            "schema": self.config["schema"], "config_sha256": digest(self.config),
            "slot_mode": self.config["slot_mode"], "duplicate_candidate_policy": self.config.get("duplicate_candidate_policy", "strict"), "candidate_ids": [r["candidate_id"] for r in rows],
            "perception_source": source, "slot_sources": contacts,
        }
        if recovery:
            state["intent_slot_pipeline"]["coverage_recovery"] = copy.deepcopy(recovery)
            state["intent_slot_pipeline"]["recovery_config"] = copy.deepcopy(self.config)
        if self.config["schema"].endswith("-v2"):
            state["intent_slot_pipeline"]["grouping"] = {
                "policy": "group_v2", "raw_count": len(json.loads(state["perception"])["candidates"]),
                "groups": [{"candidate_id": r["candidate_id"], "sources": r["sources"]} for r in raw_rows]}
        return {"state": state}
