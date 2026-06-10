# -*- coding: utf-8 -*-
"""
SOB-V6 Internal Binding and Operator Routing Probe

Goal:
  V5E showed textual policy labels (USE/PARTIAL/REJECT) do not implement U.
  V6 tests whether internal probes predict U-related outcomes:
    InternalBinding = task_probe - current_probe
    OperatorRouting = route_probe contrasts (USE/PARTIAL/ANALOGY - REJECT)

Output:
  C:\\Users\\ZH\\Desktop\\AGI\\outputs\\SOB-V6

Run:
  conda activate dhrf_4080s
  python C:\\Users\\ZH\\Desktop\\AGI\\python_script\\SOB-V6_Internal_Binding_Operator_Routing_Probe.py
"""
import os, re, json, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

EXPERIMENT_ID="SOB-V6"
OUT_DIR=Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V6"); OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SPECS={
 "qwen":{"model_name":"Qwen2.5-1.5B-Instruct","path":r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main","trust_remote_code":True},
 "llama":{"model_name":"Llama-3.2-1B-Instruct","path":r"D:\model\Llama-3.2-1B-Instruct","trust_remote_code":True},
 "gemma":{"model_name":"gemma-2-2b-it","path":r"D:\model\gemma-2-2b-it","trust_remote_code":True},
}
MODELS_TO_RUN=["qwen","llama","gemma"]
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DTYPE=torch.float16 if torch.cuda.is_available() else torch.float32
MAX_INPUT_LEN=1500; MAX_NEW_TOKENS=220; TOPK=64; N_SPLITS=5; SEED=42
ROUTES=["USE","PARTIAL_USE","ANALOGY_USE","REJECT_USE"]
RELEVANCE_WEIGHT={"high":1.0,"medium":0.55,"low":0.10}
SUBJECT_STATES=[
 ("S0",0,"symbol_only","You have only seen the name or symbol. You do not know definition, calculation, geometry, edge cases, or applications."),
 ("S1",1,"definition_known","You know only the formal definition and a few basic words. You cannot reliably calculate, reason geometrically, or transfer."),
 ("S2",2,"calculation_ability","You know definition and standard procedures. Geometry, edge cases, and transfer are limited."),
 ("S3",3,"geometric_understanding","You know definition, procedures, geometry, and edge cases. Cross-domain transfer is not yet expert-level."),
 ("S4",4,"cross_domain_transfer","You understand the object as a transferable structural tool across domains."),
]
OBJECTS={
 "derivative":{
  "name":"derivative","symbol":["d/dx","f'","prime","derivative","differentiation"],
  "core":["limit","instantaneous","rate of change","tangent","slope","local linear","differentiable"],
  "use":["local change","instantaneous change","slope","rate","sensitivity","marginal","gradient"],
  "visibility":"Derivative makes local change visible: instantaneous rate, tangent slope, local linear approximation, sensitivity, and marginal change.",
  "route":"identify the changing quantity, variable of change, and local rate/slope",
  "triads":[("change_rate","A vehicle position changes over time. Explain instantaneous velocity.","A product metric changes when one design parameter is adjusted. Explain when local sensitivity helps.","Explain the historical causes of a revolution."),("optimization","A loss function changes with a parameter. Explain why derivative information can guide an update.","A business studies marginal cost as output changes. Explain when derivative-like reasoning helps.","Analyze the theme of a poem.")]
 },
 "integral":{
  "name":"integral","symbol":["integral","∫","dx","antiderivative"],
  "core":["Riemann sum","area","accumulation","partition","antiderivative","convergence"],
  "use":["accumulation","total amount","area under","sum over","expected value","aggregate"],
  "visibility":"Integral makes accumulation visible: continuous summation, total amount, area under a curve, expected value, and aggregate effect.",
  "route":"identify what is accumulated, over what domain, and what total quantity is produced",
  "triads":[("accumulation","Velocity is known over time. Explain total distance.","A project has changing daily costs. Explain when accumulation helps estimate total cost.","Analyze a legal argument about responsibility."),("probability","A probability density is given. Explain total probability.","A researcher combines many small effects over time. Explain whether accumulation helps.","Describe the visual style of a painting.")]
 },
 "gradient":{
  "name":"gradient","symbol":["gradient","∇","nabla","partial derivative"],
  "core":["partial derivative","directional derivative","steepest ascent","level set","critical point"],
  "use":["steepest","direction of change","optimization","loss landscape","update","sensitivity"],
  "visibility":"Gradient makes direction-of-change visible: steepest increase/decrease, level set normal, optimization direction, and variable sensitivity.",
  "route":"identify the scalar quantity, controllable variables, and direction of fastest change",
  "triads":[("optimization","A model must minimize a loss. Explain how gradient indicates an update direction.","A policy has adjustable variables. Explain when gradient-like directional pressure helps.","Explain how to make a soup."),("landscape","You are on a height landscape. Explain steepest ascent or descent.","A team allocates resources among levers. Explain when directional sensitivity helps.","Analyze a fictional character's motivation.")]
 },
 "eigenvector":{
  "name":"eigenvector","symbol":["eigenvector","eigenvalue","lambda","Av"],
  "core":["invariant direction","linear transformation","scale","matrix","characteristic"],
  "use":["invariant direction","stable direction","principal component","mode","spectral","unchanged direction"],
  "visibility":"Eigenvector makes invariant direction visible: a transformation changes magnitude but preserves direction, revealing stable modes or principal axes.",
  "route":"identify the transformation, invariant directions, and stable or principal modes",
  "triads":[("stable_mode","A linear dynamical system repeats a transformation. Explain stable modes.","An organization changes but some strategic direction remains stable. Explain whether eigenvector is a useful analogy.","Explain how to roast vegetables."),("principal_axis","Data vary in many directions. Explain principal components.","A network has repeated influence propagation. Explain when eigenvector centrality helps.","Analyze rhyme in a poem.")]
 },
 "manifold":{
  "name":"manifold","symbol":["manifold","chart","atlas","surface"],
  "core":["locally Euclidean","coordinate","tangent space","curvature","geodesic","local patch"],
  "use":["local patch","global structure","latent space","state space","configuration space","local coordinates"],
  "visibility":"Manifold makes local-global structure visible: local coordinate patches connected into a global space with curvature and constraints.",
  "route":"identify locally simple patches, global structure, and constraints connecting them",
  "triads":[("latent_structure","Data lie on a lower-dimensional latent structure. Explain this.","A scientific field has local methods and global structure. Explain whether manifold is a useful analogy.","Explain why 2+2=4."),("configuration","A robot arm has constrained configurations. Explain configuration space.","An organization has local teams and global coordination. Explain whether manifold-like structure helps.","Review a comedy movie.")]
 }}

def set_seed():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def nf(x,default=np.nan):
    try: return float(x)
    except Exception: return default
def normtxt(s): return str(s).lower().replace("’","'").replace('“','"').replace('”','"')
def words(s): return re.findall(r"\b[\w'-]+\b", str(s))
def wc(s): return len(words(s))
def lexdiv(s):
    ws=[w for w in re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", normtxt(s))]
    return len(set(ws))/max(1,len(ws)) if ws else 0.0
def hits(text,markers):
    t=normtxt(text); return sorted(set([m for m in markers if normtxt(m) in t]))
def mscore(text,markers):
    h=hits(text,markers); return len(h)/max(1,len(markers)),h
def compq(text):
    n=wc(text)
    if n<=0: return 0.0
    if 25<=n<=120: return 1.0
    if n<25: return max(0.0,n/25.0)
    return max(0.0,1.0-(n-120)/260.0)
def softmax(x):
    a=np.asarray(x,dtype=np.float64); a=a-np.nanmax(a); p=np.exp(a); return p/(np.nansum(p)+1e-12)
def ent(p):
    p=np.asarray(p,dtype=np.float64); p=p/(np.nansum(p)+1e-12); return float(-np.nansum(p*np.log2(p+1e-12)))
def cos(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64); d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>1e-12 else 0.0
def topk(vals,k):
    vals=np.asarray(vals); kk=min(k,len(vals)); idx=np.argpartition(-vals,kk-1)[:kk]; return idx[np.argsort(-vals[idx])]
def center(ids,vals,W):
    if len(ids)==0: return np.zeros(W.shape[1])
    p=softmax(vals); return (W[np.asarray(ids)]*p[:,None]).sum(0)
def spread(ids,W):
    if len(ids)<=1: return 0.0
    E=W[np.asarray(ids)]; c=E.mean(0,keepdims=True); return float(np.mean(np.linalg.norm(E-c,axis=1)))
def numcols(df,cols): return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
def prefixcols(df,prefixes): return [c for c in df.columns if any(c.startswith(p) for p in prefixes) and pd.api.types.is_numeric_dtype(df[c])]

def route_for_rel(rel): return {"high":"USE","medium":"PARTIAL_USE","low":"REJECT_USE"}[rel]
def current_prompt(obj,profile):
    return f"Internal probe only. Do not answer.\nLearner state:\n{profile}\nObject: {obj['name']}.\nVisibility grounding:\n{obj['visibility']}\nEnd with READY."
def task_probe(obj,profile,rel,tid,task):
    return f"Internal binding probe only. Do not answer.\nLearner state:\n{profile}\nObject: {obj['name']}.\nVisibility grounding:\n{obj['visibility']}\nRelevance: {rel}\nTask triad: {tid}\nFuture task:\n{task}\nSilently determine whether the object binds to the task. End with READY."
def route_probe(obj,profile,route,rel,tid,task):
    if route=="USE": rt=f"Route candidate = USE. Main route: {obj['route']}."
    elif route=="PARTIAL_USE": rt=f"Route candidate = PARTIAL_USE. Use {obj['name']} only where it compresses the task."
    elif route=="ANALOGY_USE": rt=f"Route candidate = ANALOGY_USE. Use {obj['name']} only as a bounded analogy."
    else: rt=f"Route candidate = REJECT_USE. Do not use {obj['name']} if it is irrelevant."
    return f"Internal operator-routing probe only. Do not answer.\nLearner state:\n{profile}\nObject: {obj['name']}.\nVisibility grounding:\n{obj['visibility']}\nRelevance: {rel}\nTask triad: {tid}\nFuture task:\n{task}\n{rt}\nEnd with READY."
def answer_prompt(obj,profile,rel,tid,task):
    return f"You are participating in an internal binding and operator-routing audit.\nLearner state:\n{profile}\nStudied object: {obj['name']}.\nVisibility grounding:\n{obj['visibility']}\nRelevance: {rel}\nTask triad: {tid}\nFuture task:\n{task}\nAnswer concisely. Use the studied object only if it genuinely helps compress the task. If not relevant, say so and answer without forcing it."
def build_datasets():
    probes=[]; answers=[]
    for ok,obj in OBJECTS.items():
      for sid,rank,sname,profile in SUBJECT_STATES:
        base={"object_key":ok,"object_name":obj["name"],"state_id":sid,"state_rank":rank,"state_name":sname,"low_state":int(sid in LOW_STATES)}
        probes.append({"probe_row_id":len(probes),**base,"probe_type":"current","route":"NONE","triad_id":"NONE","relevance":"none","relevance_weight":0.0,"future_task":"","prompt":current_prompt(obj,profile)})
        for tid,hi,me,lo in obj["triads"]:
          for rel,task in [("high",hi),("medium",me),("low",lo)]:
            common={**base,"triad_id":tid,"relevance":rel,"relevance_weight":RELEVANCE_WEIGHT[rel],"future_task":task}
            probes.append({"probe_row_id":len(probes),**common,"probe_type":"task","route":"NONE","prompt":task_probe(obj,profile,rel,tid,task)})
            for route in ROUTES:
                probes.append({"probe_row_id":len(probes),**common,"probe_type":"route","route":route,"prompt":route_probe(obj,profile,route,rel,tid,task)})
            answers.append({"answer_row_id":len(answers),**common,"target_route":route_for_rel(rel),"prompt":answer_prompt(obj,profile,rel,tid,task)})
    return pd.DataFrame(probes), pd.DataFrame(answers)

def load_model(path,trust_remote_code=True):
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=trust_remote_code,local_files_only=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=trust_remote_code,local_files_only=True,torch_dtype=DTYPE,device_map=None)
    model.to(DEVICE); model.eval(); return model,tok
def fmt(tok,prompt):
    if hasattr(tok,"apply_chat_template") and tok.chat_template:
        try: return tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
        except Exception: return prompt
    return prompt
@torch.no_grad()
def forward_h(model,tok,prompt):
    enc=tok(fmt(tok,prompt),return_tensors="pt",truncation=True,max_length=MAX_INPUT_LEN); enc={k:v.to(DEVICE) for k,v in enc.items()}
    out=model(**enc,output_hidden_states=True,use_cache=False)
    return [h[:,-1,:].float().detach().cpu().numpy()[0] for h in out.hidden_states]
@torch.no_grad()
def gen(model,tok,prompt):
    enc=tok(fmt(tok,prompt),return_tensors="pt",truncation=True,max_length=MAX_INPUT_LEN); enc={k:v.to(DEVICE) for k,v in enc.items()}
    out=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True).strip()
def cat_ids(tok):
    out={}
    for ok,obj in OBJECTS.items():
      for cat in ["symbol","core","use"]:
        ids=[]
        for term in obj[cat]:
          for v in [term," "+term,"\n"+term]:
            t=tok(v,add_special_tokens=False).input_ids
            if t: ids.append(t[-1])
        out[f"{ok}_{cat}"]=sorted(set(ids))
    return out
def span(a,b,n): return [max(0,min(n-1,i+1)) for i in range(max(0,a),max(a,b)+1)]
def windows(mkey,n):
    nl=n-1; rel=lambda p:int(round(nl*p))
    if mkey=="qwen": return {"init":span(0,6,n),"middle":span(7,19,n),"commit":span(23,25,n),"final":span(27,27,n) if nl>=27 else span(nl,nl,n)}
    if mkey=="llama": return {"init":span(0,2,n),"middle":span(rel(.25),rel(.62),n),"commit":span(rel(.76),rel(.92),n),"final":span(nl,nl,n)}
    if mkey=="gemma": return {"init":span(0,2,n),"middle":span(7,17,n) if nl>=18 else span(rel(.25),rel(.65),n),"commit":span(21,23,n) if nl>=24 else span(rel(.76),rel(.92),n),"final":span(nl,n-1,n)}
    return {"init":span(0,rel(.2),n),"middle":span(rel(.25),rel(.65),n),"commit":span(rel(.8),rel(.92),n),"final":span(nl,nl,n)}
def feat(row,model,tok,W,cids,mkey):
    hs=forward_h(model,tok,row["prompt"]); win=windows(mkey,len(hs)); ok=row["object_key"]; F={"n_layers":len(hs)-1}
    for wn,idxs in win.items():
      seq={"symbol":[],"core":[],"use":[]}; norms=[]; abss=[]; ents=[]; sp=[]; cents=[]
      for hi in idxs:
        h=hs[hi]; logits=h@W.T; norms.append(float(np.linalg.norm(h))); abss.append(float(np.mean(np.abs(h))))
        for cat in seq:
          ids=cids.get(f"{ok}_{cat}",[]); seq[cat].append(float(np.max(logits[ids])) if ids else np.nan)
        ids=topk(logits,TOPK); vals=logits[ids]; p=softmax(vals); ents.append(ent(p)); sp.append(spread(ids,W)); cents.append(center(ids,vals,W))
      cats=list(seq.keys()); mat=np.array([[seq[c][i] for c in cats] for i in range(len(idxs))],dtype=np.float64); probs=[]
      for rs in mat:
        finite=np.isfinite(rs); probs.append(np.ones(len(cats))/len(cats) if finite.sum()==0 else softmax(np.where(finite,rs,np.nanmin(rs[finite])-10)))
      mp=np.array(probs).mean(0)
      for cat in cats: F[f"{wn}_{cat}_mass"]=float(mp[cats.index(cat)])
      F[f"{wn}_use_minus_symbol"]=F[f"{wn}_use_mass"]-F[f"{wn}_symbol_mass"]
      F[f"{wn}_hidden_norm"]=float(np.mean(norms)); F[f"{wn}_hidden_abs"]=float(np.mean(abss)); F[f"{wn}_topk_entropy"]=float(np.mean(ents)); F[f"{wn}_topk_spread"]=float(np.mean(sp))
      F[f"{wn}_topk_drift"]=float(np.mean([1-cos(cents[i],cents[i+1]) for i in range(len(cents)-1)])) if len(cents)>=2 else 0.0
    F["V_use"]=float(np.nanmean([F.get("commit_use_mass",np.nan),F.get("middle_use_mass",np.nan)])); F["V_core"]=float(np.nanmean([F.get("commit_core_mass",np.nan),F.get("middle_core_mass",np.nan)]))
    return F

def score(row):
    text=str(row.get("answer_text","")); obj=OBJECTS[row["object_key"]]; rel=row["relevance"]
    obj_terms=obj["symbol"]+obj["core"]+obj["use"]; os,_=mscore(text,obj_terms); cs,_=mscore(text,obj["core"]); us,uh=mscore(text,obj["use"]); gs,_=mscore(text,["because","therefore","structure","constraint","mapping","principle","helps","not useful","not relevant","only if","not the right tool"])
    div=lexdiv(text); comp=compq(text); neg=hits(text,["not useful","not relevant","do not force","does not help","not the right tool","limited relevance","not appropriate"])
    objuse=max(0,min(1,.55*us+.25*cs+.1*gs+.1*div))
    if rel=="low":
      nonuse=min(1,len(neg)/2) if neg else max(0,1-objuse); useful=0.0; over=objuse*(.85 if not neg else .30); corrected=.15+.75*nonuse; align=nonuse
    elif rel=="medium":
      nonuse=0.0; useful=.50*objuse+.25*gs+.15*comp+.1*div; over=.1*max(0,objuse-.7); corrected=useful-over; align=.45+.55*objuse
    else:
      nonuse=0.0; useful=.58*objuse+.18*gs+.14*comp+.1*div; over=0.0; corrected=useful; align=objuse
    useful=max(0,min(1,useful)); over=max(0,min(1,over)); corrected=max(0,min(1,corrected)); align=max(0,min(1,align))
    if rel=="low": cbit=.55*nonuse+.2*gs+.15*comp+.1*div-.5*over
    else: cbit=.52*useful+.2*align+.13*gs+.1*comp+.05*div-.25*over
    cbit=max(0,min(1,cbit)); tq=max(0,min(1,.38*corrected+.22*objuse+.2*align+.1*gs+.1*comp-.25*over)); aq=max(0,min(1,.38*corrected+.22*align+.18*gs+.12*comp+.1*div-.25*over))
    return {"object_use_score":objuse,"use_binary":int(corrected>=.45),"correct_nonuse_binary":int(nonuse>=.45),"future_success":int(cbit>=.48 and align>=.45),"relevance_alignment":align,"corrected_use_score":corrected,"useful_use_Cbit":useful,"correct_nonuse_Cbit":nonuse,"overuse_penalty":over,"Cbit_future_proxy":cbit,"TQ_future_proxy":tq,"AQ_future_proxy":aq,"use_hits":"; ".join(uh),"negation_hits":"; ".join(neg)}

def score_answers(df):
    rows=[]
    for _,r in df.iterrows():
        d=r.to_dict(); d.update(score(d)); rows.append(d)
    return pd.DataFrame(rows)

def assemble(probes,answers):
    key_task=["model_key","object_key","state_id","triad_id","relevance"]
    feat_cols=[c for c in probes.columns if pd.api.types.is_numeric_dtype(probes[c]) and c not in ["probe_row_id","state_rank","low_state","relevance_weight"]]
    cur=probes[probes.probe_type=="current"][["model_key","object_key","state_id"]+feat_cols].rename(columns={c:f"cur_{c}" for c in feat_cols})
    task=probes[probes.probe_type=="task"][key_task+feat_cols].rename(columns={c:f"task_{c}" for c in feat_cols})
    df=answers.merge(cur,on=["model_key","object_key","state_id"],how="left").merge(task,on=key_task,how="left")
    for route in ROUTES:
      rr=probes[(probes.probe_type=="route")&(probes.route==route)][key_task+feat_cols].rename(columns={c:f"route_{route}_{c}" for c in feat_cols})
      df=df.merge(rr,on=key_task,how="left")
    for c in feat_cols:
      if f"task_{c}" in df and f"cur_{c}" in df: df[f"bind_delta_{c}"]=df[f"task_{c}"]-df[f"cur_{c}"]
    for route in ["USE","PARTIAL_USE","ANALOGY_USE"]:
      for c in feat_cols:
        a=f"route_{route}_{c}"; b=f"route_REJECT_USE_{c}"
        if a in df and b in df: df[f"contrast_{route}_minus_REJECT_{c}"]=df[a]-df[b]
    return df

def splits(df,y=None,stratify=False):
    n=len(df)
    if "object_key" in df and df.object_key.nunique()>=min(N_SPLITS,n): return list(GroupKFold(n_splits=min(N_SPLITS,df.object_key.nunique())).split(np.zeros(n), y if y is not None else np.zeros(n), groups=df.object_key.values))
    if stratify and y is not None and len(np.unique(y))>1:
        vals,cnt=np.unique(y,return_counts=True); ns=min(N_SPLITS,int(cnt.min()))
        if ns>=2: return list(StratifiedKFold(n_splits=ns,shuffle=True,random_state=SEED).split(np.zeros(n),y))
    return list(KFold(n_splits=min(N_SPLITS,n),shuffle=True,random_state=SEED).split(np.zeros(n)))
def regcv(df,cols,target):
    cols=numcols(df,cols); d=df.dropna(subset=[target]).copy()
    if len(d)<12 or not cols: return {"r2":np.nan,"corr":np.nan,"n":len(d),"n_features":len(cols)}
    X=d[cols].replace([np.inf,-np.inf],np.nan).values; y=d[target].astype(float).values; pred=np.zeros(len(d))
    for tr,te in splits(d,y,False):
      mdl=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("reg",Ridge(alpha=10.0))]); mdl.fit(X[tr],y[tr]); pred[te]=mdl.predict(X[te])
    corr=float(np.corrcoef(pred,y)[0,1]) if np.std(pred)>1e-12 and np.std(y)>1e-12 else np.nan
    return {"r2":nf(r2_score(y,pred)),"corr":corr,"n":len(d),"n_features":len(cols)}
def clfcv(df,cols,target):
    cols=numcols(df,cols); d=df.dropna(subset=[target]).copy()
    if len(d)<12 or not cols: return {"auc":np.nan,"acc":np.nan,"f1":np.nan,"n":len(d),"n_features":len(cols)}
    y=d[target].astype(int).values
    if len(np.unique(y))<2: return {"auc":np.nan,"acc":np.nan,"f1":np.nan,"n":len(d),"n_features":len(cols)}
    X=d[cols].replace([np.inf,-np.inf],np.nan).values; prob=np.zeros(len(d)); pred=np.zeros(len(d),dtype=int); used=0
    for tr,te in splits(d,y,True):
      if len(np.unique(y[tr]))<2: continue
      mdl=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("clf",LogisticRegression(max_iter=2000,class_weight="balanced",solver="lbfgs"))]); mdl.fit(X[tr],y[tr]); prob[te]=mdl.predict_proba(X[te])[:,1]; pred[te]=(prob[te]>=.5).astype(int); used+=1
    if used==0: return {"auc":np.nan,"acc":np.nan,"f1":np.nan,"n":len(d),"n_features":len(cols)}
    try: auc=float(roc_auc_score(y,prob))
    except Exception: auc=np.nan
    return {"auc":auc,"acc":nf(accuracy_score(y,pred)),"f1":nf(f1_score(y,pred)),"n":len(d),"n_features":len(cols)}
def evaluate(df):
    groups={
      "CurrentOnly":prefixcols(df,["cur_"])+["state_rank","low_state","relevance_weight"],
      "TaskOnly":prefixcols(df,["task_"])+["state_rank","low_state","relevance_weight"],
      "BindingDelta":prefixcols(df,["bind_delta_"])+["state_rank","low_state","relevance_weight"],
      "RouteProbes":prefixcols(df,["route_"])+["state_rank","low_state","relevance_weight"],
      "RouteContrasts":prefixcols(df,["contrast_"])+["state_rank","low_state","relevance_weight"],
      "AllInternal":prefixcols(df,["cur_","task_","bind_delta_","route_","contrast_"])+["state_rank","low_state","relevance_weight"],
    }
    rows=[]
    for name,cols in groups.items():
      row={"feature_group":name}
      for tgt in ["Cbit_future_proxy","useful_use_Cbit","correct_nonuse_Cbit","overuse_penalty"]:
        m=regcv(df,cols,tgt); row[f"{tgt}_r2"]=m["r2"]; row[f"{tgt}_corr"]=m["corr"]; row[f"{tgt}_n_features"]=m["n_features"]
      for tgt in ["use_binary","correct_nonuse_binary","future_success"]:
        m=clfcv(df,cols,tgt); row[f"{tgt}_auc"]=m["auc"]; row[f"{tgt}_acc"]=m["acc"]; row[f"{tgt}_f1"]=m["f1"]
      rows.append(row)
    return pd.DataFrame(rows)
def verdict(summary):
    def v(g,c):
      r=summary[summary.feature_group==g]
      return nf(r[c].iloc[0]) if len(r) and c in r else np.nan
    cur=v("CurrentOnly","Cbit_future_proxy_corr"); allc=v("AllInternal","Cbit_future_proxy_corr"); bind=v("BindingDelta","Cbit_future_proxy_corr"); route=v("RouteContrasts","useful_use_Cbit_corr"); nonuse=v("RouteContrasts","correct_nonuse_binary_auc"); useauc=v("AllInternal","use_binary_auc"); succ=v("AllInternal","future_success_auc")
    gain=allc-cur if np.isfinite(allc) and np.isfinite(cur) else np.nan
    binding=bool(np.isfinite(bind) and bind>.25); routing=bool((np.isfinite(route) and route>.25) or (np.isfinite(useauc) and useauc>.70)); non=bool(np.isfinite(nonuse) and nonuse>.65); cbit=bool(np.isfinite(allc) and allc>.35 and (not np.isfinite(gain) or gain>.05)); success=bool(np.isfinite(succ) and succ>.70)
    if cbit and routing and (binding or non or success): ver="PASS_STRONG_INTERNAL_U_PROBE"
    elif (cbit and routing) or (routing and binding): ver="PASS_INTERNAL_U_PROBE"
    elif cbit or routing or binding or non: ver="PARTIAL_INTERNAL_U_PROBE"
    else: ver="FAIL_INTERNAL_U_PROBE"
    return {"verdict":ver,"current_cbit_corr":cur,"all_internal_cbit_corr":allc,"internal_gain_over_current":gain,"binding_delta_cbit_corr":bind,"route_contrast_useful_use_corr":route,"route_contrast_correct_nonuse_auc":nonuse,"all_internal_use_auc":useauc,"all_internal_success_auc":succ,"binding_support":binding,"route_support":routing,"nonuse_support":non,"cbit_support":cbit,"success_support":success}

def run_one_model(mkey,probe_ds,answer_ds):
    spec=MODEL_SPECS[mkey]; out=OUT_DIR/mkey; out.mkdir(parents=True,exist_ok=True)
    info={"experiment_id":EXPERIMENT_ID,"model_key":mkey,"model_name":spec["model_name"],"path":spec["path"],"exists":os.path.exists(spec["path"]),"status":"pending","started_at":now()}
    if not os.path.exists(spec["path"]): info.update({"status":"missing_path","error":f"Path not found: {spec['path']}"}); return info
    model=None
    try:
      t0=time.time(); model,tok=load_model(spec["path"],spec.get("trust_remote_code",True)); W=model.get_output_embeddings().weight.detach().float().cpu().numpy(); cids=cat_ids(tok)
      prows=[]; perr=[]
      for i,row in probe_ds.iterrows():
        if i%200==0: print(f"[{mkey}] probe {i}/{len(probe_ds)}",flush=True)
        try: prows.append({**row.to_dict(),"model_key":mkey,"model_name":spec["model_name"],**feat(row,model,tok,W,cids,mkey)})
        except Exception as e: perr.append({"probe_row_id":int(row["probe_row_id"]),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
      probes=pd.DataFrame(prows); probes.to_csv(out/f"{EXPERIMENT_ID}_{mkey}_probe_features.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(perr).to_csv(out/f"{EXPERIMENT_ID}_{mkey}_probe_errors.csv",index=False,encoding="utf-8-sig")
      arows=[]; aerr=[]
      for i,row in answer_ds.iterrows():
        if i%100==0: print(f"[{mkey}] answer {i}/{len(answer_ds)}",flush=True)
        try: arows.append({**row.to_dict(),"model_key":mkey,"model_name":spec["model_name"],"answer_text":gen(model,tok,row["prompt"])})
        except Exception as e: aerr.append({"answer_row_id":int(row["answer_row_id"]),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
      answers=pd.DataFrame(arows); scored=score_answers(answers) if len(answers) else pd.DataFrame(); answers.to_csv(out/f"{EXPERIMENT_ID}_{mkey}_answers.csv",index=False,encoding="utf-8-sig"); scored.to_csv(out/f"{EXPERIMENT_ID}_{mkey}_scored_answers.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(aerr).to_csv(out/f"{EXPERIMENT_ID}_{mkey}_answer_errors.csv",index=False,encoding="utf-8-sig")
      if len(probes)==0 or len(scored)==0: info.update({"status":"empty_outputs","n_probe_rows":len(probes),"n_answer_rows":len(scored),"n_errors":len(perr)+len(aerr)}); return info
      table=assemble(probes,scored); summ=evaluate(table); ver=verdict(summ)
      table.to_csv(out/f"{EXPERIMENT_ID}_{mkey}_internal_u_table.csv",index=False,encoding="utf-8-sig"); summ.to_csv(out/f"{EXPERIMENT_ID}_{mkey}_internal_u_eval_summary.csv",index=False,encoding="utf-8-sig"); pd.DataFrame([{**{"model_key":mkey},**ver}]).to_csv(out/f"{EXPERIMENT_ID}_{mkey}_verdict_summary.csv",index=False,encoding="utf-8-sig")
      info.update({"status":"done","finished_at":now(),"runtime_sec":time.time()-t0,"n_probe_rows":len(probes),"n_answer_rows":len(scored),"n_internal_rows":len(table),"n_errors":len(perr)+len(aerr),"verdict":ver["verdict"],"metrics":ver,"outputs":{"probe_features":str(out/f"{EXPERIMENT_ID}_{mkey}_probe_features.csv"),"scored_answers":str(out/f"{EXPERIMENT_ID}_{mkey}_scored_answers.csv"),"internal_u_table":str(out/f"{EXPERIMENT_ID}_{mkey}_internal_u_table.csv"),"eval_summary":str(out/f"{EXPERIMENT_ID}_{mkey}_internal_u_eval_summary.csv"),"verdict_summary":str(out/f"{EXPERIMENT_ID}_{mkey}_verdict_summary.csv")}})
      with open(out/f"{EXPERIMENT_ID}_{mkey}_verdict.json","w",encoding="utf-8") as f: json.dump(info,f,ensure_ascii=False,indent=2)
    except Exception as e: info.update({"status":"failed","finished_at":now(),"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()})
    finally:
      try:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
      except Exception: pass
    return info

def cross_summary(infos):
    rows=[]
    for inf in infos:
      m=inf.get("metrics",{}) or {}; rows.append({"model_key":inf.get("model_key"),"model_name":inf.get("model_name"),"status":inf.get("status"),"verdict":inf.get("verdict"),"current_cbit_corr":m.get("current_cbit_corr"),"all_internal_cbit_corr":m.get("all_internal_cbit_corr"),"internal_gain_over_current":m.get("internal_gain_over_current"),"binding_delta_cbit_corr":m.get("binding_delta_cbit_corr"),"route_contrast_useful_use_corr":m.get("route_contrast_useful_use_corr"),"route_contrast_correct_nonuse_auc":m.get("route_contrast_correct_nonuse_auc"),"all_internal_use_auc":m.get("all_internal_use_auc"),"all_internal_success_auc":m.get("all_internal_success_auc"),"binding_support":m.get("binding_support"),"route_support":m.get("route_support"),"nonuse_support":m.get("nonuse_support"),"cbit_support":m.get("cbit_support"),"success_support":m.get("success_support"),"n_probe_rows":inf.get("n_probe_rows"),"n_answer_rows":inf.get("n_answer_rows"),"n_internal_rows":inf.get("n_internal_rows"),"n_errors":inf.get("n_errors"),"error":inf.get("error")})
    return pd.DataFrame(rows)
def global_verdict(cross):
    done=cross[cross.status=="done"] if "status" in cross else pd.DataFrame()
    if len(done)==0: return "FAIL_SOBV6_NO_MODELS"
    vs=done.verdict.astype(str).tolist(); ps=sum(v.startswith("PASS") for v in vs); pt=sum(v.startswith("PARTIAL") for v in vs); strong=sum(v.startswith("PASS_STRONG") for v in vs)
    if strong==len(done): return "PASS_STRONG_SOBV6_CROSS_MODEL"
    if ps==len(done): return "PASS_SOBV6_CROSS_MODEL"
    if ps+pt==len(done): return "PARTIAL_SOBV6_CROSS_MODEL"
    if ps>=2: return "MIXED_POSITIVE_SOBV6"
    if ps>=1: return "MIXED_SOBV6"
    return "FAIL_SOBV6"

def main():
    set_seed(); print("="*80); print(f"[{EXPERIMENT_ID}] Internal Binding and Operator Routing Probe"); print(f"Device: {DEVICE}, dtype: {DTYPE}"); print(f"Output: {OUT_DIR}"); print("="*80)
    probe_ds,answer_ds=build_datasets(); probe_path=OUT_DIR/f"{EXPERIMENT_ID}_probe_dataset.csv"; answer_path=OUT_DIR/f"{EXPERIMENT_ID}_answer_dataset.csv"; probe_ds.to_csv(probe_path,index=False,encoding="utf-8-sig"); answer_ds.to_csv(answer_path,index=False,encoding="utf-8-sig")
    config={"experiment_id":EXPERIMENT_ID,"created_at":now(),"out_dir":str(OUT_DIR),"device":DEVICE,"dtype":str(DTYPE),"models_to_run":MODELS_TO_RUN,"model_specs":MODEL_SPECS,"routes":ROUTES,"relevance_weight":RELEVANCE_WEIGHT,"max_input_len":MAX_INPUT_LEN,"max_new_tokens":MAX_NEW_TOKENS,"topk":TOPK,"random_seed":SEED}
    with open(OUT_DIR/f"{EXPERIMENT_ID}_config.json","w",encoding="utf-8") as f: json.dump(config,f,ensure_ascii=False,indent=2)
    infos=[]
    for mkey in MODELS_TO_RUN:
      print("="*80); print(f"[{EXPERIMENT_ID}] Running model: {mkey}"); info=run_one_model(mkey,probe_ds,answer_ds); infos.append(info); print(json.dumps({"model_key":info.get("model_key"),"status":info.get("status"),"verdict":info.get("verdict"),"metrics":info.get("metrics"),"error":info.get("error")},ensure_ascii=False,indent=2))
    cross=cross_summary(infos); cross_path=OUT_DIR/f"{EXPERIMENT_ID}_cross_model_summary.csv"; cross.to_csv(cross_path,index=False,encoding="utf-8-sig"); gv=global_verdict(cross)
    global_obj={"experiment_id":EXPERIMENT_ID,"verdict":gv,"n_models":len(infos),"n_models_done":int((cross.status=="done").sum()) if "status" in cross else 0,"model_infos":infos,"outputs":{"probe_dataset":str(probe_path),"answer_dataset":str(answer_path),"cross_model_summary":str(cross_path),"out_dir":str(OUT_DIR)},"interpretation":{"InternalBinding":"task probe minus current probe features","OperatorRouting":"route probe contrasts such as USE minus REJECT","U_internal":"InternalBinding * OperatorRouting * BoundaryExecution, not textual USE policy"}}
    with open(OUT_DIR/f"{EXPERIMENT_ID}_verdict.json","w",encoding="utf-8") as f: json.dump(global_obj,f,ensure_ascii=False,indent=2)
    print("="*80); print(json.dumps(global_obj,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
