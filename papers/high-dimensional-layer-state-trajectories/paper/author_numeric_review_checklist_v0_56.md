# Final numerical review checklist

**Status: PENDING HUMAN SIGN-OFF.** This checklist is a preparation record, not evidence that an author review has already occurred. Do not add the associated author-review statement to the manuscript until H.Z. completes the checks below and signs this record.

## Review protocol

1. Compare every reported numerical value in the manuscript, figure legends, Extended Data and Supplementary Methods with its archived Source Data row or recomputation receipt.
2. Run the claim-family verifiers from `code/extension_v0_56/` and record their commits and output hashes.
3. Check that every stated conclusion matches the bound recorded in `multiplicity_claim_family_register_v0_56.md`.
4. Confirm that no final draft text upgrades a negative, nominal or descriptive result to a general claim.

## Required item families

| Item family | Archive / verifier | Reviewer initials and date |
|---|---|---|
| Main coordinate audit and projection controls | Global Source Data manifest and main-figure source tables | |
| ΔU construction, leakage controls and functional-window boundary | Supplementary Source Data and Tables 1-4 | |
| Geometry, random-initialization and reordered-execution controls | Main/Extended Data geometry tables | |
| Transition closure and target-excluded forecast | `source_data/extension_v0_54/` and its verifier | |
| Relation-graph actuator, operator ablations and output guards | Main actuator Source Data | |
| Mixed-arithmetic transfer boundary | `code/extension_v0_55/verify_extension_v0_55.py` | |
| v0.56 predecision coordinate competition | `verify_predecision_coordinate_audit.py` | |
| v0.56 empirical direction null | `verify_empirical_direction_null.py` | |
| v0.56 new-graph geometry | `verify_newgraph_geometry_confirmation.py` | |
| Data/code availability links, version labels and repository hashes | Submission release manifest | |

## Sign-off

I, H.Z., confirm that I have personally completed the item-by-item comparison above against the archived Source Data and recorded any discrepancies before final submission.

Name: Heng Zhao
Signature / initials: ____________________
Date: ____________________
Release commit or archive version: ____________________

After signature, the manuscript may add: "H.Z. performed a final item-by-item review of all reported numerical results against the archived Source Data; the checks used for this review are included in the code archive."
