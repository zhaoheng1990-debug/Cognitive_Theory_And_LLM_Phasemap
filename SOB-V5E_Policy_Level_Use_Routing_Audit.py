# -*- coding: utf-8 -*-
r"""
SOB-V5E: Policy-Level Use Routing Audit

This experiment tests whether the U term in
    Meaning(S,O,T)=V(S,O)*R(O,T)*U(S,O,T)
is better treated as a policy-level routing variable than as a natural-language
instruction to "use the object".

Output:
    C:\Users\ZH\Desktop\AGI\outputs\SOB-V5E
"""
import os, re, json, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT_ID = "SOB-V5E"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V5E")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "qwen": {"model_name":"Qwen2.5-1.5B-Instruct", "path":r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main", "trust_remote_code":True},
    "llama": {"model_name":"Llama-3.2-1B-Instruct", "path":r"D:\model\Llama-3.2-1B-Instruct", "trust_remote_code":True},
    "gemma": {"model_name":"gemma-2-2b-it", "path":r"D:\model\gemma-2-2b-it", "trust_remote_code":True},
}
MODELS_TO_RUN = ["qwen", "llama", "gemma"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_INPUT_LEN = 1600
MAX_NEW_TOKENS = 220
SEED = 42

RELEVANCE_WEIGHT = {"high":1.0, "medium":0.55, "low":0.10}
LOW_STATES = {"S0","S1","S2"}
CONDITIONS = ["control", "oracle_policy", "noisy_policy", "wrong_policy", "free_boundary_policy"]

SUBJECT_STATES = [
    ("S0",0,"symbol_only","You have only seen the name or symbol. You do not know definition, calculation, geometry, edge cases, or applications."),
    ("S1",1,"definition_known","You know only the formal definition and a few basic words. You cannot reliably calculate, reason geometrically, or transfer."),
    ("S2",2,"calculation_ability","You know definition and standard procedures. Geometry, edge cases, and transfer are limited."),
    ("S3",3,"geometric_understanding","You know definition, procedures, geometry, and edge cases. Cross-domain transfer is not yet expert-level."),
    ("S4",4,"cross_domain_transfer","You understand the object as a transferable structural tool across domains."),
]

OBJECTS = {
 "derivative": {"name":"derivative", "terms":{"symbol":["d/dx","f'","prime","derivative","differentiation"],"core":["limit","instantaneous","rate of change","tangent","slope","local linear","differentiable"],"use":["local change","instantaneous change","slope","rate","sensitivity","marginal","gradient"]}, "visibility":"Derivative makes local change visible: instantaneous rate, tangent slope, local linear approximation, sensitivity, and marginal change.", "route":"Use derivative by identifying the changing quantity, the variable of change, and the local rate/slope.", "triads":[("change_rate","A vehicle position changes over time. Explain instantaneous velocity.","A product metric changes when one design parameter is adjusted. Explain when local sensitivity helps.","Explain the historical causes of a revolution."),("optimization","A loss function changes with a parameter. Explain why derivative information can guide an update.","A business studies marginal cost as output changes. Explain when derivative-like reasoning helps.","Analyze the theme of a poem.")]},
 "integral": {"name":"integral", "terms":{"symbol":["integral","∫","dx","antiderivative"],"core":["Riemann sum","area","accumulation","partition","antiderivative","convergence"],"use":["accumulation","total amount","area under","sum over","expected value","aggregate"]}, "visibility":"Integral makes accumulation visible: continuous summation, total amount, area under a curve, expected value, and aggregate effect.", "route":"Use integral by identifying what is accumulated, over what domain, and what total quantity is produced.", "triads":[("accumulation","Velocity is known over time. Explain total distance.","A project has changing daily costs. Explain when accumulation helps estimate total cost.","Analyze a legal argument about responsibility."),("probability","A probability density is given. Explain total probability.","A researcher combines many small effects over time. Explain whether accumulation helps.","Describe the visual style of a painting.")]},
 "gradient": {"name":"gradient", "terms":{"symbol":["gradient","∇","nabla","partial derivative"],"core":["partial derivative","directional derivative","steepest ascent","level set","critical point"],"use":["steepest","direction of change","optimization","loss landscape","update","sensitivity"]}, "visibility":"Gradient makes direction-of-change visible: steepest increase/decrease, level set normal, optimization direction, and variable sensitivity.", "route":"Use gradient by identifying the scalar quantity, the controllable variables, and the direction of fastest change.", "triads":[("optimization","A model must minimize a loss. Explain how gradient indicates an update direction.","A policy has adjustable variables. Explain when gradient-like directional pressure helps.","Explain how to make a soup."),("landscape","You are on a height landscape. Explain steepest ascent or descent.","A team allocates resources among levers. Explain when directional sensitivity helps.","Analyze a fictional character's motivation.")]},
 "eigenvector": {"name":"eigenvector", "terms":{"symbol":["eigenvector","eigenvalue","lambda","Av"],"core":["invariant direction","linear transformation","scale","matrix","characteristic"],"use":["invariant direction","stable direction","principal component","mode","spectral","unchanged direction"]}, "visibility":"Eigenvector makes invariant direction visible: a transformation changes magnitude but preserves direction, revealing stable modes or principal axes.", "route":"Use eigenvector by identifying the transformation, invariant directions, and the stable/principal mode.", "triads":[("stable_mode","A linear dynamical system repeats a transformation. Explain stable modes.","An organization changes but some strategic direction remains stable. Explain whether eigenvector is a useful analogy.","Explain how to roast vegetables."),("principal_axis","Data vary in many directions. Explain principal components.","A network has repeated influence propagation. Explain when eigenvector centrality helps.","Analyze rhyme in a poem.")]},
 "manifold": {"name":"manifold", "terms":{"symbol":["manifold","chart","atlas","surface"],"core":["locally Euclidean","coordinate","tangent space","curvature","geodesic","local patch"],"use":["local patch","global structure","latent space","state space","configuration space","local coordinates"]}, "visibility":"Manifold makes local-global structure visible: local coordinate patches connected into a global space with curvature and constraints.", "route":"Use manifold by identifying the locally simple patches, the global structure, and the constraints connecting them.", "triads":[("latent_structure","Data lie on a lower-dimensional latent structure. Explain this.","A scientific field has local methods and global structure. Explain whether manifold is a useful analogy.","Explain why 2+2=4."),("configuration","A robot arm has constrained configurations. Explain configuration space.","An organization has local teams and global coordination. Explain whether manifold-like structure helps.","Review a comedy movie.")]},
}

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default

def norm(s): return str(s).lower().replace("’", "'").replace("“", '"').replace("”", '"')
def words(s): return len(re.findall(r"\b[\w'-]+\b", str(s)))
def lexdiv(s):
    ws = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", norm(s))
    return len(set(ws))/max(1,len(ws)) if ws else 0.0

def hits(text, markers):
    t=norm(text); return sorted(set(m for m in markers if norm(m) in t))
def score_markers(text, markers):
    h=hits(text, markers); return len(h)/max(1,len(markers)), h

def compq(text):
    n=words(text)
    if n<=0: return 0.0
    if 25<=n<=120: return 1.0
    if n<25: return max(0.0,n/25.0)
    return max(0.0,1.0-(n-120)/260.0)

def policy_for(condition, relevance, triad_id):
    if condition=="control": return "NONE"
    if condition=="oracle_policy": return {"high":"USE","medium":"PARTIAL_USE","low":"REJECT_USE"}[relevance]
    if condition=="wrong_policy": return {"high":"REJECT_USE","medium":"REJECT_USE","low":"USE"}[relevance]
    if condition=="noisy_policy":
        if relevance=="high": return "USE"
        if relevance=="medium": return "ANALOGY_USE" if triad_id in {"stable_mode","latent_structure"} else "PARTIAL_USE"
        return "PARTIAL_USE" if triad_id in {"change_rate","accumulation"} else "REJECT_USE"
    if condition=="free_boundary_policy": return "MODEL_SELECT"
    return "NONE"

def policy_instruction(obj, condition, relevance, triad_id):
    p=policy_for(condition,relevance,triad_id)
    if p=="NONE": return ""
    if p=="USE": return f"Routing policy = USE. Use {obj['name']} as the main organizing object. Route: {obj['route']}"
    if p=="PARTIAL_USE": return f"Routing policy = PARTIAL_USE. Use {obj['name']} only for the part of the task where it gives compression; do not overextend it. Route: {obj['route']}"
    if p=="ANALOGY_USE": return f"Routing policy = ANALOGY_USE. Use {obj['name']} only as an analogy or structural lens, and explicitly mark its boundary."
    if p=="REJECT_USE": return f"Routing policy = REJECT_USE. State that {obj['name']} is not the right tool for this task, then answer without forcing it."
    if p=="MODEL_SELECT": return f"Choose one routing policy before answering: USE, PARTIAL_USE, ANALOGY_USE, or REJECT_USE. Use {obj['name']} only if it genuinely compresses the task. Route if used: {obj['route']}"
    return ""

def make_prompt(obj, profile, condition, relevance, triad_id, task):
    instr=policy_instruction(obj,condition,relevance,triad_id)
    return f"""You are participating in a policy-level use-routing causal audit.

Learner state:
{profile}

Studied object: {obj['name']}.
Visibility grounding:
{obj['visibility']}

Condition: {condition}
Relevance: {relevance}
Task triad: {triad_id}
Future task:
{task}

{instr}

Answer concisely. Focus on whether the studied object helps compress the task."""

def build_dataset():
    rows=[]
    for obj_key,obj in OBJECTS.items():
        for state_id,rank,state_name,profile in SUBJECT_STATES:
            for triad_id,high,medium,low in obj["triads"]:
                for rel,task in [("high",high),("medium",medium),("low",low)]:
                    for cond in CONDITIONS:
                        rows.append({"row_id":len(rows),"object_key":obj_key,"object_name":obj["name"],"state_id":state_id,"state_rank":rank,"state_name":state_name,"low_state":int(state_id in LOW_STATES),"triad_id":triad_id,"relevance":rel,"relevance_weight":RELEVANCE_WEIGHT[rel],"condition":cond,"policy_label":policy_for(cond,rel,triad_id),"future_task":task,"prompt":make_prompt(obj,profile,cond,rel,triad_id,task)})
    return pd.DataFrame(rows)

def load_model(path, trust_remote_code=True):
    tok=AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True, torch_dtype=DTYPE, device_map=None)
    model.to(DEVICE); model.eval()
    return model,tok

def chat(tok,prompt):
    if hasattr(tok,"apply_chat_template") and tok.chat_template:
        try: return tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
        except Exception: return prompt
    return prompt

@torch.no_grad()
def gen(model,tok,prompt):
    text=chat(tok,prompt)
    enc=tok(text,return_tensors="pt",truncation=True,max_length=MAX_INPUT_LEN)
    enc={k:v.to(DEVICE) for k,v in enc.items()}
    out=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score_answer(row):
    text=str(row.get("answer_text","")); obj=OBJECTS[row["object_key"]]; rel=row["relevance"]
    core_s,_=score_markers(text,obj["terms"]["core"])
    use_s,use_hits=score_markers(text,obj["terms"]["use"])
    obj_s,obj_hits=score_markers(text,obj["terms"]["symbol"]+obj["terms"]["core"]+obj["terms"]["use"])
    gen_s,_=score_markers(text,["because","therefore","structure","constraint","mapping","principle","helps","not useful","not relevant","only if","not the right tool"])
    neg=hits(text,["not useful","not relevant","do not force","does not help","not the right tool","limited relevance","not appropriate"])
    div=lexdiv(text); comp=compq(text)
    object_use=max(0.0,min(1.0,0.55*use_s+0.25*core_s+0.10*gen_s+0.10*div))
    if rel=="low":
        correct_nonuse=min(1.0,len(neg)/2.0) if neg else max(0.0,1.0-object_use)
        useful=0.0
        over=object_use*(0.85 if not neg else 0.30)
        corrected=0.15+0.75*correct_nonuse
        align=correct_nonuse
    elif rel=="medium":
        correct_nonuse=0.0
        useful=0.50*object_use+0.25*gen_s+0.15*comp+0.10*div
        over=0.10*max(0.0,object_use-0.70)
        corrected=useful-over
        align=0.45+0.55*object_use
    else:
        correct_nonuse=0.0
        useful=0.58*object_use+0.18*gen_s+0.14*comp+0.10*div
        over=0.0
        corrected=useful
        align=object_use
    useful=max(0,min(1,useful)); over=max(0,min(1,over)); corrected=max(0,min(1,corrected)); align=max(0,min(1,align))
    if rel=="low": cbit=0.55*correct_nonuse+0.20*gen_s+0.15*comp+0.10*div-0.50*over
    else: cbit=0.52*useful+0.20*align+0.13*gen_s+0.10*comp+0.05*div-0.25*over
    tq=0.38*corrected+0.22*object_use+0.20*align+0.10*gen_s+0.10*comp-0.25*over
    aq=0.38*corrected+0.22*align+0.18*gen_s+0.12*comp+0.10*div-0.25*over
    cbit=max(0,min(1,cbit)); tq=max(0,min(1,tq)); aq=max(0,min(1,aq))
    return {"future_word_count":words(text),"object_marker_score":obj_s,"use_marker_score":use_s,"object_hits":"; ".join(obj_hits),"use_hits":"; ".join(use_hits),"negation_hits":"; ".join(neg),"object_use_score":object_use,"relevance_alignment":align,"corrected_use_score":corrected,"useful_use_Cbit":useful,"correct_nonuse_Cbit":correct_nonuse,"overuse_penalty":over,"Cbit_future_proxy":cbit,"TQ_future_proxy":tq,"AQ_future_proxy":aq,"future_success":int(cbit>=0.48 and align>=0.45)}

def score_df(df):
    out=[]
    for _,r in df.iterrows():
        d=r.to_dict(); d.update(score_answer(d)); out.append(d)
    return pd.DataFrame(out)

def paired_deltas(df):
    keys=["model_key","object_key","state_id","triad_id","relevance"]
    metrics=["Cbit_future_proxy","TQ_future_proxy","AQ_future_proxy","useful_use_Cbit","correct_nonuse_Cbit","overuse_penalty","object_use_score","corrected_use_score","relevance_alignment","future_success","low_state","state_rank","relevance_weight"]
    ctrl=df[df.condition=="control"][keys+metrics].rename(columns={c:f"{c}_control" for c in metrics})
    pairs=[]
    for cond in ["oracle_policy","noisy_policy","wrong_policy","free_boundary_policy"]:
        sub=df[df.condition==cond][keys+metrics].rename(columns={c:f"{c}_intervention" for c in metrics})
        p=ctrl.merge(sub,on=keys,how="inner"); p["condition"]=cond
        for m in metrics: p[f"delta_{m}"]=p[f"{m}_intervention"]-p[f"{m}_control"]
        p["low_state"]=p["low_state_control"]; p["state_rank"]=p["state_rank_control"]; p["relevance_weight"]=p["relevance_weight_control"]
        pairs.append(p)
    return pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()

def summarize(pair):
    groups=[("all",pair)]
    for cond in ["oracle_policy","noisy_policy","wrong_policy","free_boundary_policy"]:
        groups += [(cond,pair[pair.condition==cond]),(f"{cond}_low_states",pair[(pair.condition==cond)&(pair.low_state==1)])]
        for rel in ["high","medium","low"]:
            groups.append((f"{cond}_{rel}",pair[(pair.condition==cond)&(pair.relevance==rel)]))
            groups.append((f"{cond}_low_states_{rel}",pair[(pair.condition==cond)&(pair.low_state==1)&(pair.relevance==rel)]))
    metrics=["delta_Cbit_future_proxy","delta_TQ_future_proxy","delta_AQ_future_proxy","delta_useful_use_Cbit","delta_correct_nonuse_Cbit","delta_overuse_penalty","delta_object_use_score","delta_corrected_use_score","delta_relevance_alignment","delta_future_success"]
    rows=[]
    for name,sub in groups:
        row={"group":name,"n":len(sub)}
        for m in metrics:
            vals=pd.to_numeric(sub[m],errors="coerce").dropna() if m in sub.columns else pd.Series(dtype=float)
            row[f"{m}_mean"]=safe_float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_positive_rate"]=safe_float((vals>0).mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def verdict(summary):
    def g(group,metric):
        r=summary[summary.group==group]
        return safe_float(r[f"{metric}_mean"].iloc[0]) if len(r) and f"{metric}_mean" in r.columns else np.nan
    oh=g("oracle_policy_low_states_high","delta_useful_use_Cbit"); om=g("oracle_policy_low_states_medium","delta_useful_use_Cbit")
    ol_non=g("oracle_policy_low_states_low","delta_correct_nonuse_Cbit"); ol_over=g("oracle_policy_low_states_low","delta_overuse_penalty")
    ohc=g("oracle_policy_low_states_high","delta_Cbit_future_proxy"); omc=g("oracle_policy_low_states_medium","delta_Cbit_future_proxy"); olc=g("oracle_policy_low_states_low","delta_Cbit_future_proxy")
    fh=g("free_boundary_policy_low_states_high","delta_useful_use_Cbit"); flo=g("free_boundary_policy_low_states_low","delta_overuse_penalty")
    whc=g("wrong_policy_low_states_high","delta_Cbit_future_proxy"); wlo=g("wrong_policy_low_states_low","delta_overuse_penalty")
    oracle_use=bool((np.isfinite(oh) and oh>0.02) or (np.isfinite(om) and om>0.02))
    oracle_cbit=bool((np.isfinite(ohc) and ohc>0.02) or (np.isfinite(omc) and omc>0.02))
    oracle_safe=bool((not np.isfinite(ol_over) or ol_over<0.02) and (not np.isfinite(ol_non) or ol_non>=-0.02))
    wrong_neg=bool((np.isfinite(whc) and whc<0.01) or (np.isfinite(wlo) and wlo>0.02))
    free_sup=bool((np.isfinite(fh) and fh>0.0) and (not np.isfinite(flo) or flo<0.03))
    if oracle_use and oracle_cbit and oracle_safe and wrong_neg: v="PASS_STRONG_POLICY_LEVEL_USE_ROUTING"
    elif oracle_use and oracle_safe: v="PASS_POLICY_LEVEL_USE_ROUTING"
    elif oracle_use or oracle_cbit or free_sup: v="PARTIAL_POLICY_LEVEL_USE_ROUTING"
    else: v="FAIL_POLICY_LEVEL_USE_ROUTING"
    return {"verdict":v,"oracle_high_useful_use_delta":oh,"oracle_medium_useful_use_delta":om,"oracle_low_correct_nonuse_delta":ol_non,"oracle_low_overuse_delta":ol_over,"oracle_high_cbit_delta":ohc,"oracle_medium_cbit_delta":omc,"oracle_low_cbit_delta":olc,"free_high_useful_use_delta":fh,"free_low_overuse_delta":flo,"wrong_high_cbit_delta":whc,"wrong_low_overuse_delta":wlo,"oracle_use_support":oracle_use,"oracle_cbit_support":oracle_cbit,"oracle_safety_support":oracle_safe,"free_boundary_support":free_sup,"wrong_negative_control":wrong_neg}

def run_one_model(model_key,dataset):
    spec=MODEL_SPECS[model_key]; model_out=OUT_DIR/model_key; model_out.mkdir(parents=True,exist_ok=True)
    info={"experiment_id":EXPERIMENT_ID,"model_key":model_key,"model_name":spec["model_name"],"path":spec["path"],"exists":os.path.exists(spec["path"]),"status":"pending","started_at":now()}
    if not os.path.exists(spec["path"]):
        info.update({"status":"missing_path","error":f"Path not found: {spec['path']}"}); return info
    model=None
    try:
        t0=time.time(); model,tok=load_model(spec["path"], spec.get("trust_remote_code",True))
        rows=[]; errors=[]
        for i,r in dataset.iterrows():
            if i%100==0: print(f"[{model_key}] row {i}/{len(dataset)}",flush=True)
            try: rows.append({**r.to_dict(),"model_key":model_key,"model_name":spec["model_name"],"answer_text":gen(model,tok,r["prompt"])})
            except Exception as e: errors.append({"row_id":int(r["row_id"]),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
        answers=pd.DataFrame(rows); scored=score_df(answers) if not answers.empty else pd.DataFrame()
        answers.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_future_answers.csv",index=False,encoding="utf-8-sig")
        scored.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_future_scored_answers.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(errors).to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_errors.csv",index=False,encoding="utf-8-sig")
        if scored.empty:
            info.update({"status":"empty_outputs","n_future_rows":0,"n_errors":len(errors)}); return info
        pair=paired_deltas(scored); summ=summarize(pair); vd=verdict(summ)
        pair.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_paired_deltas.csv",index=False,encoding="utf-8-sig")
        summ.to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_effect_summary.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame([{**{"model_key":model_key},**vd}]).to_csv(model_out/f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv",index=False,encoding="utf-8-sig")
        info.update({"status":"done","finished_at":now(),"runtime_sec":time.time()-t0,"n_future_rows":len(scored),"n_pairs":len(pair),"n_errors":len(errors),"verdict":vd.get("verdict"),"metrics":vd,"outputs":{"future_answers":str(model_out/f"{EXPERIMENT_ID}_{model_key}_future_answers.csv"),"future_scored_answers":str(model_out/f"{EXPERIMENT_ID}_{model_key}_future_scored_answers.csv"),"paired_deltas":str(model_out/f"{EXPERIMENT_ID}_{model_key}_paired_deltas.csv"),"effect_summary":str(model_out/f"{EXPERIMENT_ID}_{model_key}_effect_summary.csv"),"verdict_summary":str(model_out/f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv")}})
        with open(model_out/f"{EXPERIMENT_ID}_{model_key}_verdict.json","w",encoding="utf-8") as f: json.dump(info,f,ensure_ascii=False,indent=2)
    except Exception as e:
        info.update({"status":"failed","finished_at":now(),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
    finally:
        try:
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
    return info

def cross_model_summary(infos):
    rows=[]
    for info in infos:
        m=info.get("metrics",{}) or {}
        rows.append({"model_key":info.get("model_key"),"model_name":info.get("model_name"),"status":info.get("status"),"verdict":info.get("verdict"),"oracle_high_useful_use_delta":m.get("oracle_high_useful_use_delta"),"oracle_medium_useful_use_delta":m.get("oracle_medium_useful_use_delta"),"oracle_low_correct_nonuse_delta":m.get("oracle_low_correct_nonuse_delta"),"oracle_low_overuse_delta":m.get("oracle_low_overuse_delta"),"oracle_high_cbit_delta":m.get("oracle_high_cbit_delta"),"oracle_medium_cbit_delta":m.get("oracle_medium_cbit_delta"),"oracle_low_cbit_delta":m.get("oracle_low_cbit_delta"),"free_high_useful_use_delta":m.get("free_high_useful_use_delta"),"free_low_overuse_delta":m.get("free_low_overuse_delta"),"wrong_high_cbit_delta":m.get("wrong_high_cbit_delta"),"wrong_low_overuse_delta":m.get("wrong_low_overuse_delta"),"oracle_use_support":m.get("oracle_use_support"),"oracle_cbit_support":m.get("oracle_cbit_support"),"oracle_safety_support":m.get("oracle_safety_support"),"free_boundary_support":m.get("free_boundary_support"),"wrong_negative_control":m.get("wrong_negative_control"),"n_future_rows":info.get("n_future_rows"),"n_pairs":info.get("n_pairs"),"n_errors":info.get("n_errors"),"error":info.get("error")})
    return pd.DataFrame(rows)

def global_verdict(cross):
    done=cross[cross.status=="done"] if "status" in cross.columns else pd.DataFrame()
    if len(done)==0: return "FAIL_SOBV5E_NO_MODELS"
    vs=done.verdict.astype(str).tolist(); ps=sum(v.startswith("PASS") for v in vs); pss=sum(v.startswith("PASS_STRONG") for v in vs); part=sum(v.startswith("PARTIAL") for v in vs)
    if pss==len(done): return "PASS_STRONG_SOBV5E_CROSS_MODEL"
    if ps==len(done): return "PASS_SOBV5E_CROSS_MODEL"
    if ps+part==len(done): return "PARTIAL_SOBV5E_CROSS_MODEL"
    if ps>=2: return "MIXED_POSITIVE_SOBV5E"
    if ps>=1: return "MIXED_SOBV5E"
    return "FAIL_SOBV5E"

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    print("="*80); print(f"[{EXPERIMENT_ID}] Policy-Level Use Routing Audit"); print(f"Device: {DEVICE}, dtype: {DTYPE}"); print(f"Output: {OUT_DIR}"); print("="*80)
    ds=build_dataset(); ds_path=OUT_DIR/f"{EXPERIMENT_ID}_future_dataset.csv"; ds.to_csv(ds_path,index=False,encoding="utf-8-sig")
    config={"experiment_id":EXPERIMENT_ID,"created_at":now(),"out_dir":str(OUT_DIR),"device":DEVICE,"dtype":str(DTYPE),"models_to_run":MODELS_TO_RUN,"model_specs":MODEL_SPECS,"conditions":CONDITIONS,"relevance_weight":RELEVANCE_WEIGHT,"max_input_len":MAX_INPUT_LEN,"max_new_tokens":MAX_NEW_TOKENS,"random_seed":SEED}
    with open(OUT_DIR/f"{EXPERIMENT_ID}_config.json","w",encoding="utf-8") as f: json.dump(config,f,ensure_ascii=False,indent=2)
    infos=[]
    for mk in MODELS_TO_RUN:
        print("="*80); print(f"[{EXPERIMENT_ID}] Running model: {mk}")
        info=run_one_model(mk,ds); infos.append(info)
        print(json.dumps({"model_key":info.get("model_key"),"status":info.get("status"),"verdict":info.get("verdict"),"metrics":info.get("metrics"),"error":info.get("error")},ensure_ascii=False,indent=2))
    cross=cross_model_summary(infos); cross_path=OUT_DIR/f"{EXPERIMENT_ID}_cross_model_summary.csv"; cross.to_csv(cross_path,index=False,encoding="utf-8-sig")
    gv=global_verdict(cross)
    global_obj={"experiment_id":EXPERIMENT_ID,"verdict":gv,"n_models":len(infos),"n_models_done":int((cross.status=="done").sum()) if "status" in cross.columns else 0,"model_infos":infos,"outputs":{"future_dataset":str(ds_path),"cross_model_summary":str(cross_path),"out_dir":str(OUT_DIR)},"interpretation":{"oracle_policy":"high->USE, medium->PARTIAL_USE, low->REJECT_USE.","noisy_policy":"partially perturbed policy to test robustness.","wrong_policy":"negative control: relevant->reject, low->use.","free_boundary_policy":"model selects route label before answering.","PASS_STRONG_POLICY_LEVEL_USE_ROUTING":"Oracle policy improves relevant useful Cbit, preserves low-relevance safety, and wrong policy is negative."}}
    with open(OUT_DIR/f"{EXPERIMENT_ID}_verdict.json","w",encoding="utf-8") as f: json.dump(global_obj,f,ensure_ascii=False,indent=2)
    print("="*80); print(json.dumps(global_obj,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
