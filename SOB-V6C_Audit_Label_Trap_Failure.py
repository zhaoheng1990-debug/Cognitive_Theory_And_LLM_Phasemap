# -*- coding: utf-8 -*-
"""
SOB-V6C.Audit: Label Distribution and Trap Failure Audit
Reads SOB-V6C outputs and diagnoses whether PARTIAL result comes from label degeneration,
marker misses, overuse-label collapse, or genuine trap-boundary failure.
No model forward. No CLI args.
"""
import json, re, time
from pathlib import Path
import numpy as np
import pandas as pd

EXPERIMENT_ID = "SOB-V6C_Audit"
SCRIPT_VERSION = "SOB-V6C_AUDIT_LABEL_TRAP_FAILURE_v1"
IN_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V6C")
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V6C_Audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS = ["qwen", "llama", "gemma"]

OBJECTS = {
    "derivative": {"wrong_object":"integral", "object_terms":["d/dx","prime","derivative","differentiation","limit","instantaneous","rate of change","tangent","slope","local linear","differentiable","local change","rate","sensitivity","marginal"], "wrong_terms":["integral","accumulation","area","sum","total amount"]},
    "integral": {"wrong_object":"gradient", "object_terms":["integral","antiderivative","riemann sum","area","accumulation","partition","convergence","total amount","area under","sum over","expected value","aggregate"], "wrong_terms":["gradient","steepest","direction","optimization","loss landscape"]},
    "gradient": {"wrong_object":"eigenvector", "object_terms":["gradient","nabla","partial derivative","directional derivative","steepest ascent","level set","critical point","steepest","direction of change","optimization","loss landscape","update","sensitivity"], "wrong_terms":["eigenvector","eigenvalue","invariant direction","principal component","stable mode"]},
    "eigenvector": {"wrong_object":"manifold", "object_terms":["eigenvector","eigenvalue","lambda","invariant direction","linear transformation","scale","matrix","characteristic","stable direction","principal component","mode","spectral","unchanged direction"], "wrong_terms":["manifold","chart","local patch","curvature","geodesic","configuration space"]},
    "manifold": {"wrong_object":"derivative", "object_terms":["manifold","chart","atlas","surface","locally euclidean","coordinate","tangent space","curvature","geodesic","local patch","global structure","latent space","state space","configuration space","local coordinates"], "wrong_terms":["derivative","rate","slope","tangent","local change","marginal"]},
}

EXPLICIT_REJECT_PATTERNS = [
    "not useful","not relevant","not the right tool","not appropriate","not structurally relevant",
    "does not help","do not force","avoid forcing","should not use","would not use",
    "isn't useful","isn't relevant","wrong tool","limited relevance","no need to use","unnecessary",
    "not a mathematical","not a math","do not use","avoid using","should avoid"
]
SOFT_REJECT_PATTERNS = [
    "instead","rather than","more natural","better tool","more appropriate","ordinary explanation",
    "without forcing","metaphor","rhetoric","cultural","historical","legal","social",
    "in this context","not literally","not mathematically"
]
OVERUSE_PATTERNS = [
    "use the derivative","using the derivative","derivative shows","derivative can","derivative helps",
    "use the integral","using the integral","integral shows","integral can","integral helps",
    "use the gradient","using the gradient","gradient shows","gradient can","gradient helps",
    "use the eigenvector","using the eigenvector","eigenvector shows","eigenvector can","eigenvector helps",
    "use the manifold","using the manifold","manifold shows","manifold can","manifold helps",
]

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def norm(s):
    return str(s).lower().replace("’", "'").replace("“", '"').replace("”", '"')

def marker_hits(text, pats):
    t = norm(text)
    return sorted(set([p for p in pats if norm(p) in t]))

def describe_binary(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n":0,"n_pos":0,"n_neg":0,"pos_rate":np.nan,"degenerate":True}
    vals = (s > 0).astype(int)
    n = int(len(vals)); n_pos = int(vals.sum()); n_neg = n - n_pos
    return {"n":n,"n_pos":n_pos,"n_neg":n_neg,"pos_rate":n_pos/max(1,n),"degenerate":bool(n_pos==0 or n_neg==0)}

def summarize_labels(df, model_key, version):
    groups = [("all", df)]
    for trap in sorted(df.get("trap_type", pd.Series(dtype=str)).dropna().unique()):
        groups.append((f"trap={trap}", df[df["trap_type"] == trap]))
    for obj in sorted(df.get("object_key", pd.Series(dtype=str)).dropna().unique()):
        groups.append((f"object={obj}", df[df["object_key"] == obj]))
    for trap in sorted(df.get("trap_type", pd.Series(dtype=str)).dropna().unique()):
        for obj in sorted(df.get("object_key", pd.Series(dtype=str)).dropna().unique()):
            sub = df[(df["trap_type"] == trap) & (df["object_key"] == obj)]
            if len(sub): groups.append((f"trap={trap}|object={obj}", sub))
    rows=[]
    for name, sub in groups:
        row={"model_key":model_key,"scoring_version":version,"group":name,"n":int(len(sub))}
        for col in ["correct_nonuse_binary","overuse_binary","boundary_success"]:
            if col in sub.columns:
                d=describe_binary(sub[col])
                row[f"{col}_n"]=d["n"]; row[f"{col}_pos"]=d["n_pos"]; row[f"{col}_neg"]=d["n_neg"]
                row[f"{col}_pos_rate"]=d["pos_rate"]; row[f"{col}_degenerate"]=d["degenerate"]
        for col in ["correct_nonuse_Cbit","overuse_penalty","overuse_severity","Cbit_future_proxy","object_use_score","explicit_reject_score"]:
            if col in sub.columns:
                vals=pd.to_numeric(sub[col], errors="coerce").dropna()
                row[f"{col}_mean"]=float(vals.mean()) if len(vals) else np.nan
                row[f"{col}_std"]=float(vals.std()) if len(vals)>1 else np.nan
                row[f"{col}_min"]=float(vals.min()) if len(vals) else np.nan
                row[f"{col}_max"]=float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def rescore(df, model_key):
    rows=[]
    for _, r in df.iterrows():
        text=str(r.get("answer_text", ""))
        obj_key=str(r.get("object_key", ""))
        meta=OBJECTS.get(obj_key, {"object_terms":[],"wrong_terms":[],"wrong_object":""})
        explicit_hits=marker_hits(text, EXPLICIT_REJECT_PATTERNS)
        soft_hits=marker_hits(text, SOFT_REJECT_PATTERNS)
        overuse_hits=marker_hits(text, OVERUSE_PATTERNS)
        object_hits=marker_hits(text, meta["object_terms"])
        wrong_hits=marker_hits(text, meta["wrong_terms"] + [meta.get("wrong_object", "")])
        explicit=min(1.0, len(explicit_hits)/2.0)
        soft=min(1.0, len(soft_hits)/2.0)
        object_use=min(1.0, len(object_hits)/4.0)
        wrong_use=min(1.0, len(wrong_hits)/3.0)
        overuse=min(1.0, 0.70*object_use + 0.30*min(1.0, len(overuse_hits)/2.0))
        if r.get("trap_type") == "wrong_object_trap":
            nonuse=max(explicit, 0.50*soft + 0.50*max(0.0, wrong_use-object_use))
        else:
            nonuse=max(explicit, soft, max(0.0, 1.0-object_use))
        success=int(nonuse >= 0.45 and overuse < 0.45)
        row=r.to_dict()
        row.update({
            "model_key":model_key,
            "explicit_reject_hits_v2":"; ".join(explicit_hits),
            "soft_reject_hits_v2":"; ".join(soft_hits),
            "overuse_hits_v2":"; ".join(overuse_hits),
            "object_hits_v2":"; ".join(object_hits),
            "wrong_hits_v2":"; ".join(wrong_hits),
            "explicit_reject_v2":explicit,
            "soft_reject_v2":soft,
            "object_use_v2":object_use,
            "wrong_use_v2":wrong_use,
            "overuse_v2":overuse,
            "correct_nonuse_v2":nonuse,
            "correct_nonuse_binary_v2":int(nonuse>=0.45),
            "overuse_binary_v2":int(overuse>=0.45),
            "boundary_success_v2":success,
            "delta_correct_nonuse_v2_minus_old":nonuse-float(r.get("correct_nonuse_Cbit",0.0)),
            "delta_overuse_v2_minus_old":overuse-float(r.get("overuse_severity",0.0)),
            "delta_success_v2_minus_old":success-int(r.get("boundary_success",0)),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def failure_examples(df, top_n=50):
    ex=[]
    if "boundary_success" in df.columns:
        a=df[(df["boundary_success"]==0)&(df["boundary_success_v2"]==1)].copy(); a["case_type"]="old_fail_v2_success_marker_miss"; ex.append(a)
        b=df[(df["boundary_success"]==1)&(df["boundary_success_v2"]==0)].copy(); b["case_type"]="old_success_v2_fail_possible_overuse"; ex.append(b)
    if "object_use_score" in df.columns:
        c=df[(pd.to_numeric(df["object_use_score"], errors="coerce")>0.35)&(df["explicit_reject_v2"]>=0.5)].copy(); c["case_type"]="reject_but_object_markers_high"; ex.append(c)
    d=df[(df["trap_type"]=="wrong_object_trap")&(df.get("boundary_success",0)==0)].copy(); d["case_type"]="wrong_object_trap_failure_old"; ex.append(d)
    if not ex: return pd.DataFrame()
    out=pd.concat(ex, ignore_index=True)
    keep=["model_key","case_type","object_key","state_id","trap_type","correct_nonuse_Cbit","overuse_severity","boundary_success","correct_nonuse_v2","overuse_v2","boundary_success_v2","negation_hits","alternative_hits","explicit_reject_hits_v2","soft_reject_hits_v2","object_hits_v2","wrong_hits_v2","answer_text"]
    keep=[c for c in keep if c in out.columns]
    return out[keep].head(top_n)

def audit_one_model(model_key):
    model_dir=IN_DIR / model_key
    scored_path=model_dir / f"SOB-V6C_{model_key}_scored_answers.csv"
    info={"model_key":model_key,"scored_path":str(scored_path),"exists_scored":scored_path.exists(),"status":"pending"}
    if not scored_path.exists():
        info["status"]="missing_scored"; return info, None, None, None
    scored=pd.read_csv(scored_path)
    rescored=rescore(scored, model_key)
    old_summary=summarize_labels(scored, model_key, "old")
    v2tmp=rescored.rename(columns={"correct_nonuse_binary_v2":"correct_nonuse_binary","overuse_binary_v2":"overuse_binary","boundary_success_v2":"boundary_success","correct_nonuse_v2":"correct_nonuse_Cbit","overuse_v2":"overuse_severity"})
    v2_summary=summarize_labels(v2tmp, model_key, "v2_marker_rescore")
    label_summary=pd.concat([old_summary, v2_summary], ignore_index=True)
    examples=failure_examples(rescored)
    old_all=old_summary[old_summary["group"]=="all"].iloc[0].to_dict() if len(old_summary[old_summary["group"]=="all"]) else {}
    v2_all=v2_summary[v2_summary["group"]=="all"].iloc[0].to_dict() if len(v2_summary[v2_summary["group"]=="all"]) else {}
    def g(d,k): return d.get(k, np.nan)
    info.update({
        "status":"done",
        "n_rows":int(len(scored)),
        "old_correct_nonuse_pos_rate":g(old_all,"correct_nonuse_binary_pos_rate"),
        "old_overuse_pos_rate":g(old_all,"overuse_binary_pos_rate"),
        "old_boundary_success_pos_rate":g(old_all,"boundary_success_pos_rate"),
        "v2_correct_nonuse_pos_rate":g(v2_all,"correct_nonuse_binary_pos_rate"),
        "v2_overuse_pos_rate":g(v2_all,"overuse_binary_pos_rate"),
        "v2_boundary_success_pos_rate":g(v2_all,"boundary_success_pos_rate"),
        "old_overuse_degenerate":bool(g(old_all,"overuse_binary_degenerate")),
        "v2_overuse_degenerate":bool(g(v2_all,"overuse_binary_degenerate")),
        "old_boundary_success_degenerate":bool(g(old_all,"boundary_success_degenerate")),
        "v2_boundary_success_degenerate":bool(g(v2_all,"boundary_success_degenerate")),
        "n_marker_miss_cases":int(len(rescored[(rescored.get("boundary_success",0)==0)&(rescored.get("boundary_success_v2",0)==1)])),
        "n_possible_overuse_cases":int(len(rescored[(rescored.get("boundary_success",0)==1)&(rescored.get("boundary_success_v2",0)==0)])),
        "n_wrong_trap_fail_old":int(len(rescored[(rescored["trap_type"]=="wrong_object_trap")&(rescored.get("boundary_success",0)==0)])),
        "n_wrong_trap_fail_v2":int(len(rescored[(rescored["trap_type"]=="wrong_object_trap")&(rescored.get("boundary_success_v2",0)==0)])),
    })
    return info, rescored, label_summary, examples

def global_diagnosis(cross):
    diagnosis=[]
    if len(cross)==0: return ["no_model_outputs"]
    if cross["old_overuse_degenerate"].fillna(False).all(): diagnosis.append("old_overuse_binary_degenerate_all_models")
    elif cross["old_overuse_degenerate"].fillna(False).any(): diagnosis.append("old_overuse_binary_degenerate_some_models")
    if cross["v2_overuse_degenerate"].fillna(False).all(): diagnosis.append("v2_overuse_still_degenerate_all_models")
    elif cross["v2_overuse_degenerate"].fillna(False).any(): diagnosis.append("v2_overuse_still_degenerate_some_models")
    if pd.to_numeric(cross["n_marker_miss_cases"], errors="coerce").mean() >= 5: diagnosis.append("substantial_marker_miss_or_scoring_underestimate")
    if pd.to_numeric(cross["n_wrong_trap_fail_old"], errors="coerce").mean() >= 10: diagnosis.append("wrong_object_trap_is_real_failure_or_scoring_gap")
    rates=pd.to_numeric(cross.get("old_boundary_success_pos_rate", pd.Series(dtype=float)), errors="coerce")
    if rates.notna().any() and ((rates<0.10)|(rates>0.90)).any(): diagnosis.append("boundary_success_class_imbalance_detected")
    if not diagnosis: diagnosis.append("no_major_label_degeneracy_detected")
    return diagnosis

def main():
    print(f"script_version={SCRIPT_VERSION}")
    print(f"input_dir={IN_DIR}")
    print(f"output_dir={OUT_DIR}")
    infos=[]; labels=[]; examples=[]
    for model_key in MODELS:
        model_out=OUT_DIR/model_key; model_out.mkdir(parents=True, exist_ok=True)
        info, rescored, label_summary, ex = audit_one_model(model_key)
        infos.append(info)
        if rescored is not None:
            rescored.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_rescored_answers_v2.csv", index=False, encoding="utf-8-sig")
        if label_summary is not None:
            label_summary.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_label_distribution_summary.csv", index=False, encoding="utf-8-sig")
            labels.append(label_summary)
        if ex is not None and len(ex):
            ex.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_failure_examples.csv", index=False, encoding="utf-8-sig")
            examples.append(ex)
        with open(model_out/f"{EXPERIMENT_ID}_{model_key}_audit.json", "w", encoding="utf-8") as f: json.dump(info, f, ensure_ascii=False, indent=2)
    cross=pd.DataFrame(infos)
    cross_path=OUT_DIR/f"{EXPERIMENT_ID}_cross_model_audit_summary.csv"
    cross.to_csv(cross_path, index=False, encoding="utf-8-sig")
    if labels: pd.concat(labels, ignore_index=True).to_csv(OUT_DIR/f"{EXPERIMENT_ID}_all_label_distribution_summary.csv", index=False, encoding="utf-8-sig")
    if examples: pd.concat(examples, ignore_index=True).to_csv(OUT_DIR/f"{EXPERIMENT_ID}_all_failure_examples.csv", index=False, encoding="utf-8-sig")
    diagnosis=global_diagnosis(cross)
    verdict="AUDIT_COMPLETED"
    if any("degenerate" in d for d in diagnosis): verdict="AUDIT_LABEL_DEGENERATION_CONFIRMED"
    if "substantial_marker_miss_or_scoring_underestimate" in diagnosis: verdict="AUDIT_SCORING_UNDERESTIMATE_CONFIRMED"
    obj={"experiment_id":EXPERIMENT_ID,"script_version":SCRIPT_VERSION,"verdict":verdict,"created_at":now(),"input_dir":str(IN_DIR),"out_dir":str(OUT_DIR),"diagnosis":diagnosis,"model_infos":infos,"outputs":{"cross_model_audit_summary":str(cross_path),"all_label_distribution_summary":str(OUT_DIR/f'{EXPERIMENT_ID}_all_label_distribution_summary.csv'),"all_failure_examples":str(OUT_DIR/f'{EXPERIMENT_ID}_all_failure_examples.csv')},"interpretation":{"old_overuse_binary_degenerate":"Explains NaN overuse AUC in V6C if true.","marker_miss_cases":"Old scoring may miss valid rejection / no-use answers.","wrong_object_trap_fail":"If persistent under v2, trap boundary is real unsolved item.","next_step":"Only design V6D after this audit identifies whether scoring or real trap-boundary failure dominates."}}
    with open(OUT_DIR/f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
    print(json.dumps(obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
