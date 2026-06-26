# HPM-RT1 v1.1 中文说明

## 第一指导基线

本项目的第一指导文件是：

```text
HPM_RT1_FIRST_GUIDANCE_zh.md
```

完整理论源文件是：

```text
HPM_series_theory_baseline_standalone_v1_0.md
```

HPM 之上的跨项目理论基线备份在：

```text
theory_baselines/Cognitive_Research_Architecture_v3_5/
```

并已安装成本地 Codex skill：

```text
cognitive-research-architecture
```

配套术语表、公式表和算子表是：

```text
HPM_concepts_and_formula_tables_v1_0.md
```

所有后续工程升级、规则修复、测试集接入、版本说明和打包输出，均默认继承：

```text
TheoryBaseline: HPM series theory baseline standalone v1.0
UpperTheoryBaseline: Cognitive Research Architecture v3.5
ConceptFormulaOperatorTables: HPM concepts and formula tables v1.0
MethodologyKernel: v1.1
```

冲突规则：若 Cognitive Research Architecture v3.5 中的 HPM 部分与当前 HPM 专用理论基线冲突，HPM 细节以当前项目 HPM 基线为准；非 HPM 的上位认知论、LLM 动力学、ACS、方法论和对应关系，以 Cognitive Research Architecture v3.5 为准。

若连续出现样例级修补、关键词堆叠或阈值反复调整，必须先执行 ObjectUpgradingAudit，再决定是否新增规则。

HPM-RT1 v1.0 的当前定位是：**LLM 黑盒外实时预警与审计层**。

它部署在大模型外部，用可见输入、可见输出、用户提供证据、页面/接口元数据来做幻觉风险预警、约束场识别、证据边界审计、建议动作、日志留痕和 PDF 报告。

它不是模型内部探针，不读取也不声称拥有以下信息：

```text
模型隐藏状态
logits / token 概率
attention
真实 chain-of-thought
模型内部置信度
服务商侧检索链路
采样参数或内部推理过程
```

v1.0 的工程边界是：

```text
黑盒外部预警
advisory only
不自动改写模型答案
不接管最终决策
不修改模型权重
不证明模型内部是否“知道”某事实
```

HPM-RT1 Beta1 是一个外置 advisory guard。它用于模型运行时的幻觉风险、约束场识别、证据需求、建议动作、跨模型 / 跨参数窗口对齐诊断、参数回写和 trace 采集。

它不做三件事：

```text
不自动改写模型答案
不接管最终决策
不修改模型权重
```

默认输出目录：

```text
C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_beta_output
```

## 0. v0.5 新增内容

v0.5 的核心升级是：所有主要说明文件均提供中文与英文双版本，方便内部研究、外部 beta 测试者、跨团队协作和未来开源说明。

```text
README_zh.md / README_en.md
MANIFEST_zh.md / MANIFEST_en.md
HPM_RT1_Beta_Module_Protocol_zh.md / HPM_RT1_Beta_Module_Protocol_en.md
HPM_RT1_Testing_and_Output_Interpretation_zh.md / HPM_RT1_Testing_and_Output_Interpretation_en.md
HPM_RT1_Testset_Construction_Guide_zh.md / HPM_RT1_Testset_Construction_Guide_en.md
HPM_RT1_W_Manifold_Mapping_Gap_Note_zh.md / HPM_RT1_W_Manifold_Mapping_Gap_Note_en.md
```

旧的无后缀说明文件仍保留，但建议测试者优先阅读带 `_zh` 或 `_en` 后缀的版本。

## 1. 推荐阅读顺序

中文测试者：

```text
HPM_RT1_FIRST_GUIDANCE_zh.md
theory_baselines/Cognitive_Research_Architecture_v3_5/
HPM_series_theory_baseline_standalone_v1_0.md
HPM_concepts_and_formula_tables_v1_0.md
README_zh.md
HPM_RT1_Beta_Module_Protocol_zh.md
HPM_RT1_Testset_Construction_Guide_zh.md
HPM_RT1_Testing_and_Output_Interpretation_zh.md
HPM_RT1_W_Manifold_Mapping_Gap_Note_zh.md
```

英文测试者：

```text
README_en.md
HPM_RT1_Beta_Module_Protocol_en.md
HPM_RT1_Testset_Construction_Guide_en.md
HPM_RT1_Testing_and_Output_Interpretation_en.md
HPM_RT1_W_Manifold_Mapping_Gap_Note_en.md
```

## 2. 快速安装与运行

在包根目录执行：

```bash
pip install -e .
```

运行基础样例：

```bash
python -m hpm_rt1_beta.cli --jsonl examples/sample_cases.jsonl --output-dir "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_beta_output"
```

运行教学测试集：

```bash
python -m hpm_rt1_beta.cli --jsonl examples/hpm_rt1_testset_case_study.jsonl --output-dir "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_v05_case_study_output"
```

运行跨模型 / 跨参数窗口对齐诊断：

```bash
python tools/hpm_rt1_window_alignment_diagnostic.py --eval-jsonl examples/sample_alignment_eval.jsonl --output-dir "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_alignment_output"
```

将诊断结果回写到 RT1 active config：

```bash
python tools/hpm_rt1_writeback_calibration.py --profile "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_alignment_output\hpm_rt1_calibration_profile.json" --install-dir . --dry-run

python tools/hpm_rt1_writeback_calibration.py --profile "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_alignment_output\hpm_rt1_calibration_profile.json" --install-dir .
```

## 3. 当前模块结构

```text
hpm_rt1_beta/core.py          主守门器
hpm_rt1_beta/sro.py           结构分辨率 / 约束场识别
hpm_rt1_beta/risk.py          风险评分
hpm_rt1_beta/advisory.py      建议动作策略
hpm_rt1_beta/adapters.py      跨模型 adapter 接口
hpm_rt1_beta/calibration.py   runtime calibration loader
hpm_rt1_beta/logger.py        JSONL trace 记录
tools/hpm_rt1_window_alignment_diagnostic.py   跨模型窗口诊断
tools/hpm_rt1_writeback_calibration.py         参数回写脚本
examples/hpm_rt1_testset_case_study.jsonl      教学测试集
examples/hpm_rt1_testset_template.csv          外部测试模板
```

## 4. Beta1 边界

RT1 v0.5 仍保持 Beta1 边界：

```text
advisory only
no answer rewrite
no model weight update
no active HPM-12 repair
```

HPM-12 candidate repair 在独立验证通过前，只能作为 shadow / candidate arm 记录，不应进入 active policy。

## 5. 跨模型 / 跨参数原则

可以跨模型挂载，但阈值不能默认共享。

```text
Wrapper interface 可以跨模型。
Calibration / thresholds / risk weights 必须按 model_family × parameter_scale × task_family × window_id 诊断。
```

这也是 v0.2 以后加入窗口对齐诊断和参数回写的原因。

## 6. W 流形测绘缺口

v0.4 起正式记录：RT1 仍缺少 W-Manifold Mapping / ManifoldCoverageRisk 的完整实现。v0.5 继续保持该边界。

当前只做：

```text
MCR shadow annotation
测试集构造提示
未来 HPM-WM1 实验接口预留
```

不做：

```text
hidden-state W mapping
TopK/VIM full probe
active MCR policy
```

## 7. 输出文件

RT1 CLI 输出通常包括：

```text
hpm_rt1_beta_trace.jsonl
```

窗口诊断输出通常包括：

```text
hpm_rt1_group_window_metrics.csv
hpm_rt1_window_alignment_summary.csv
hpm_rt1_calibration_profile.json
hpm_rt1_alignment_report.md
```

回写脚本修改：

```text
hpm_rt1_beta/config/rt1_active_config.json
```

## 8. 最小结论

v0.5 的目标不是证明 HPM 已经消除幻觉，而是提供一个可测试、可校准、可记录、可跨模型比较的 advisory guard，并为 HPM-12 independent stream 与后续 W-Manifold Mapping 实验收集规范化数据。


## v0.6 Sample Testsets

新增真实案例样本集与模板：`HPM_RT1_Sample_Testsets_zh.md` / `HPM_RT1_Sample_Testsets_en.md`，以及 `examples/hpm_rt1_external_testset_min32.jsonl`、`examples/hpm12_independent_testset_180.jsonl`、`examples/hpm_rt1_alignment_eval_min32.jsonl`、`examples/hpm12_delayed_feedback_template.csv`。


---

## v0.7 核心更新：HPM = 幻觉风险治理

v0.7 正式修正 HPM 核心命题：HPM 不是万能幻觉修正器，而是幻觉风险治理模块。新增 `AnswerabilityGate`：当 W 覆盖不足、证据不足、约束场不清、时效性不足、CGA/约束对齐低或 source-prior 冲突时，RT1 应建议 `retrieve / ask / refuse / defer / uncertainty mark`，而不是尝试把 unsupported closure 修成正确答案。

新增说明文件：

```text
HPM_Core_Theorem_Update_zh.md
HPM_Core_Theorem_Update_en.md
HPM_Core_Theorem_Update.md
```

新增运行时输出字段：

```text
answerability_gate
```

边界不变：Beta1 advisory only，不改写答案、不更新模型权重、不激活 HPM-12 candidate repair。

---

## v0.8 新增：ContextFrameGate / 上下文框架门控

v0.8 增加 `ContextFrameGate`，主链升级为：

```text
Prompt → ContextFrameGate → SRO → AnswerabilityGate → ClosureMonitor → ConstraintAlignmentAudit → PolicyAction
```

新增输出字段：

```text
context_frame_gate
risk_scores.context_frame_risk
```

核心原则：

```text
先判定语境，再判定幻觉。
Hallucination is constraint violation under the active context frame.
```

新增说明文件：

```text
HPM_ContextFrameGate_Update_zh.md
HPM_ContextFrameGate_Update_en.md
```

新增样例集：

```text
examples/hpm_rt1_context_frame_gate_cases.jsonl
examples/hpm_rt1_context_frame_gate_cases.csv
```

运行：

```bash
python -m hpm_rt1_beta.cli --jsonl examples/hpm_rt1_context_frame_gate_cases.jsonl
```

## v0.9 新增

新增 Codex 工程开发指南：`HPM_RT1_Codex_Engineering_Guide_zh.md`。该指南补充 API Server、前端审计台、工程结构、核心模块职责、调试路径、集成路线和验收标准。

---

## v1.0 定位冻结：LLM 黑盒外实时预警与审计层

v1.0 正式冻结当前工程定位：

```text
HPM-RT1 = black-box LLM external runtime warning and audit layer
```

证据坐标：

```text
Internal Project Evidence + Controlled Runtime Wrapper Evidence
```

Accepted object：

```text
可见输入 / 可见输出 / 证据边界 / 页面或接口元数据
  -> ContextFrameGate
  -> AnswerabilityGate
  -> RiskScores
  -> AdvisoryAction
  -> 过程预警 / 完成后日志
```

保留边界：

```text
不读取模型内部状态
不读取 logits、attention、隐藏层、真实推理链
不证明模型内部知识状态
不在网页端阻断模型生成
不替代模型服务侧治理
```

v1.0 能做的是黑盒外部风险治理：发送前 prompt 预检、生成中 DOM 增量预警、完成后 gateway 审计、JSONL/PDF 留痕。模型上下文适配覆盖 GPT、Gemini、Kimi、Doubao、Claude、DeepSeek、Qwen、Llama、Gemma 与本地/其他模型。
如果未来进入模型服务侧或 API 中间层，可以升级为更强的流式治理和输出前门控；但这属于下一层部署，不属于当前黑盒外部版本的能力声明。

---

## v1.1 新增：按 prompt 类型动态加权的审计置信度

v1.1 修正 `overall_audit_confidence` 的计算方式：不再把所有置信度维度简单平均，而是根据 `task_family`、`constraint_field` 和 `reality_mode` 选择不同权重。

这里的置信度是“审计判断可置信度”：值越高，表示 HPM-RT1 越有把握认为自己的审计结论成立；它不是模型回答的风险值。模型回答风险请看 `risk_scores.hallucination_risk`、`answerability_gate` 和 `advisory_action`。

```text
时效/过期记忆类：更重证据支持与新鲜度
来源冲突/低证据陷阱：更重证据支持与约束对齐
反事实/角色扮演/仿真：更重语境框架识别
紧急行动/高风险领域：更重行动安全与建议动作
创作开放任务：降低证据权重，提高语境与建议动作权重
```

每次输出会额外包含：

```text
audit_confidence_profile
audit_confidence_weights
audit_confidence_threshold
audit_confidence_gate
```

---

## PCAO 理论输入索引

当前 PCAO 路线相关文档如下：

```text
HPM_RT1_PCAO_PathLevel_Patch_Inheritance_v0_1_zh.md
HPM_RT1_PCAO_PolicyState_Inheritance_v0_2_zh.md
HPM_RT1_PCAO_Engineering_Activation_Notes_v0_2_zh.md
```

当前约定：

```text
v0.1 已完成 stage A shadow integration
v0.2 已接收为 theory input，但尚未激活为 core policy
```
