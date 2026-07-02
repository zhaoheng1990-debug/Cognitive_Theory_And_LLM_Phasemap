# Forbidden Capability Block Report 7201A7300Z

Verdict: PASS_FORBIDDEN_CAPABILITY_BLOCKED

Forbidden capabilities are denied before execution, even if a malformed dispatch envelope names them. The explicit test covers `EXTERNAL_API_MUTATION -> BLOCKED`.

Forbidden set:

- SEND_EMAIL
- WIRE_TRANSFER
- EXTERNAL_API_MUTATION
- GIT_PUSH
- PRODUCTION_DEPLOY
- GLOBAL_MEMORY_WRITE
- GLOBAL_ICM_WRITE
- UNBOUNDED_WEB_ACTION
- LEGAL_SIGNATURE
- INVESTMENT_COMMITMENT

No external mutation, production deploy, global memory write, or global ICM write was performed.
