from __future__ import annotations


def replay_truthfulness_matrix() -> list[dict[str, object]]:
    return [
        {
            "capability_block": "portable_dev_rc_runtime_subset",
            "configured_project_replay": False,
            "portable_artifact_replay": True,
            "standalone_portable_dev_rc_source_tree_replay": True,
            "production_release": False,
            "allowed_claim": "Minimal source subset can run package-local tests after extraction.",
            "forbidden_claim": "Does not prove full historical configured project replay or production release.",
        },
        {
            "capability_block": "prior_long_run_evidence",
            "configured_project_replay": True,
            "portable_artifact_replay": True,
            "standalone_portable_dev_rc_source_tree_replay": False,
            "production_release": False,
            "allowed_claim": "Prior evidence remains inspectable as artifacts when supplied.",
            "forbidden_claim": "Do not convert artifact inspection into full standalone runtime proof.",
        },
    ]
