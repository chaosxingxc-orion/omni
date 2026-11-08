# P candidate preparation

`prepare_p_candidate.py` derives a prompt-only P candidate from the pinned coverage parent recipe. It reuses that recipe's stage, prepare, register, admission, launcher, caches and assets; it is not a general-purpose recipe converter. The release must contain the pinned template and preserve the parent's protected runtime sources and image. Supply the private absolute paths and SHA-256 pins described by `example.json`.

Run `python scripts/environment/prepare_p_candidate.py --config CONFIG.json` for the default read-only preflight. It verifies pins and the release manifest without creating the target root. Add `--execute` to create a new root and run the CPU preparation and real admission stages, producing a launcher; it does not launch generation or queue a job.

`candidate.generation_name` sets the new container name only. The launcher inherits the parent recipe's `--out` path; its basename need not match the container name. Resolve `--out` through the launcher's actual mounts to locate `generation-released.json` on the host, rather than deriving a directory from the container name.

If execution fails, preserve the target root, stage argv and logs, container identity and all receipts. Diagnose the original failing stage and reconcile any running or unknown request before a recovery decision. Do not blindly retry, overwrite the root, or invent completion receipts. A preflight pass does not guarantee later stages succeed.

An optional `runtime_binding_override` selects a registered deployment through configuration. It contains `runtime_binding` (model fragment, profile, receipt, service session), `runtime_binding_adaptation` with pinned parent/current model and receipt identities, `assets` mapping workspace-relative files to absolute paths and SHA-256 pins, `serving_root`, and `guard_directory`. The existing same-model-and-request comparison still applies.

The admission recipe and generation launcher may use different original serving copies. The override maps their exact serving destination to the configured read-only serving root, sets the deployment profile, and points the generation lease and locks mount to the same guard directory. It does not change the selected supply, few-shot policy, ranking window, or running generations. A new binding requires its own registered workspace and admission; an old admission receipt is not transferable.
