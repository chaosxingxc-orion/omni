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
        source = copy.deepcopy(state["latest_perception"]["execution_receipt"]["final_contact"])
        batches = [rows] if self.config["slot_mode"] == "batched" else [[row] for row in rows]
        results, contacts = {}, []
        for batch in batches:
            state["intent_slot_context"] = {
                "candidates": batch, "perception_source": source,
                "candidate_set_sha256": digest(rows),
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
            parsed = slot_results(role["actor_output"], ids)
            results.update(parsed)
            contacts.append({"candidate_ids": ids,
                             "source": copy.deepcopy(role["execution_receipt"]["final_contact"]),
                             "result_sha256": digest(parsed)})
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
        return {"state": state}
