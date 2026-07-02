You are working in the Logos AgentOS repository.

Implement:

AgentOS_HumanReview_UI_1B_DecisionExportBridgeForPB1S_NoMutationNoAccept

Context:
- HumanReview UI-1A passed.
- UI-1A provided a readable static review surface for WRC_BOUNDARY_001.
- PM feedback is now a hard design rule: do not force human reviewers to edit raw CSV/schema as the primary workflow.
- PB1S should receive a decision record exported from the UI, not manually authored as a table-first object.

Core task:
Add a local decision export bridge:
HumanReviewObject -> readable UI card -> safe decision button -> PB1S-compatible JSON/CSV decision record -> schema validation.

Allowed:
- static local HTML/JS/CSS;
- local JSON fixtures;
- local CSV/JSON export;
- local validation tests;
- no backend mutation;
- no external API.

Forbidden:
- RetentionAccept;
- SignedAccept;
- AcceptedBaseline write;
- AcceptedEvidence write;
- MemoryUnit write;
- Operator promotion;
- Policy promotion;
- external API;
- web browsing;
- requiring PM to edit raw machine CSV/schema as the primary workflow.

Required return pack:
AgentOS_HumanReview_UI_1B_DecisionExportBridgeForPB1S_NoMutationNoAccept_Return_Pack_v0_1.zip

Required verdict:
PASS_HUMAN_REVIEW_UI_1B_DECISION_EXPORT_BRIDGE_FOR_PB1S_NO_MUTATION_NO_ACCEPT

Complete return file list:
- README.md
- human_review_ui_1b_summary.md
- review_surface_export_bridge.html
- human_review_object_schema.json
- pb_retention_review_fixture.json
- pb1s_decision_schema.json
- exported_pb1s_decision_example.json
- exported_pb1s_decision_example.csv
- decision_export_validation_report.md
- ui_to_pb1s_mapping_report.md
- usability_checklist.md
- no_csv_first_review_audit.md
- no_mutation_audit.md
- no_accept_no_promotion_audit.md
- accessibility_notes.md
- render_notes.md
- isolated_replay_report.md
- tests_summary.md
- return_files_manifest_v0_1.json
- command_replay_log.md
