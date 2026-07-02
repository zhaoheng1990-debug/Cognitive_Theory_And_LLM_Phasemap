# AgentOS CoreSlim Base / AgentOS CoreSlim 基座

## English

AgentOS CoreSlim is a lightweight, governance-first runtime base for building cross-project AgentOS systems. It is designed to provide reusable kernel policies, local execution boundaries, candidate-only evolution flows, and validation artifacts that downstream AgentOS projects can start from without inheriting unstable research-line mutations.

This branch is the clean **base-maintenance** line. It is intentionally separated from research-AgentOS self-evolution outputs. Research outputs may be used as evidence, but they are not promoted into the base unless a future PM-approved sync seed explicitly imports a stable patch.

### What AgentOS Is

AgentOS is an operating layer for governed cognitive workflows. It is not a single chatbot, document parser, workflow script, or model wrapper. Its purpose is to coordinate roles, evidence, policies, candidate changes, and human review gates so that an AgentOS project can evolve safely across domains.

In the CoreSlim base, the emphasis is on:

- bounded local execution rather than uncontrolled automation;
- kernel-owned authorization rather than tool-owned decisions;
- candidate-only evolution rather than direct mutation of accepted registries;
- replayable evidence rather than unverifiable memory claims;
- rollback and audit trails rather than silent state changes;
- cross-project portability rather than one-off project scripts.

### Current Status

- Status: base-maintenance candidate for PM / human review
- Branch: `AgentOS`
- Validation: local tests pass
- Intended use: bootstrap downstream AgentOS projects and maintain the CoreSlim base
- Not claimed: production release, global registry activation, official theory-baseline mutation, AGI achievement, or autonomous production deployment

### Recommended Usage

AgentOS CoreSlim is best used as a governed base plus one or more replaceable runners. A runner is the interactive coding or operating surface that helps a human or project maintainer execute AgentOS tasks. Recommended runner options include:

- **Codex**: useful for repository work, local code edits, tests, packaging, return packs, and GitHub handoff.
- **Claude Code**: useful as an alternate coding runner for local repo navigation, patch work, and review-style workflows.
- **WorkBuddy**: useful as an operational workspace runner when a project needs task orchestration, handoff tracking, or day-to-day execution support.

The runner is not the AgentOS authority. In the CoreSlim model, runners should be treated as operator interfaces. They may read instructions, edit files, run tests, prepare patches, and call bounded tools, but they should not directly promote candidates, mutate accepted registries, or bypass Kernel / PM / human review gates.

A typical setup is:

```text
Human / PM
  -> Runner: Codex, Claude Code, WorkBuddy, or another local operator surface
  -> AgentOS Kernel policy layer
  -> Harness execution layer(s)
  -> Receipts, hash inventories, rollback pointers, return packs
```

### Harness Execution Layers

AgentOS can connect to multiple Harness layers. A Harness is an execution adapter, not a governance owner. Different projects may attach different Harnesses depending on the task:

- local shell / Python test Harness;
- document parsing or extraction Harness;
- data validation Harness;
- browser or UI automation Harness;
- simulation or benchmark Harness;
- packaging and release Harness;
- domain-specific Harness, such as VC, education, manufacturing, research, legal, or healthcare adapters.

Harness outputs should be treated as evidence or execution receipts. They may support a candidate decision, but they do not become accepted truth by themselves. Kernel policy, evidence lineage, permission review, schema review, retention review, and human/PM authorization determine whether anything can be promoted.

### Quick Start Pattern

1. Start from the `AgentOS` branch or a verified CoreSlim return pack.
2. Choose a runner, usually Codex for repository maintenance or Claude Code / WorkBuddy for alternate local workflows.
3. Read the current seed pack, pointer, or task prompt before editing.
4. Keep all new behavior candidate-only unless explicit authorization says otherwise.
5. Run the local tests and any task-specific Harness checks.
6. Emit a return pack with manifest, hash inventory, validation report, and rollback pointer.
7. Treat pointer updates and accepted-registry updates as review candidates until approved.

### Core Capabilities

#### 1. Kernel-Bounded Tool Execution

`CodexToolBridge` provides a bounded local execution bridge. It can read artifacts, run local scripts/tests, package files, generate hash inventories, replay receipts, and roll back local writes when explicitly authorized by a kernel dispatch envelope.

Boundary:

- The bridge is a Harness Plane executor, not a final decision owner.
- It cannot authorize itself.
- It blocks forbidden capabilities such as external API mutation, production deploy, global memory write, global ICM write, git push, legal signature, investment commitment, and unbounded web action.

#### 2. Project-Scoped ICM Evolution

`AutonomousICMEvolutionPolicy` allows mature, evidence-backed candidates to become project-scoped durable artifacts, such as memory units, operator memory, policy priors, applicability gates, or quarantine records.

Boundary:

- Writes are project-scoped only.
- Global memory, production ICM, official theory baseline, and accepted evidence are not mutated.
- Rollback and replay metadata are required.
- Weak evidence stays candidate-only.
- Negative-transfer candidates route to quarantine rather than promotion.

#### 3. Baseline Evolution Proposal Protocol

`BaselineEvolutionProposalProtocol` converts mature project-scoped learning into human-reviewable baseline update proposals.

Boundary:

- It proposes S3/S4 or theory-baseline updates.
- It does not apply them.
- Signed human authorization remains required before any official/global baseline write.
- Candidates with insufficient evidence, unresolved conflicts, or high negative-transfer risk are deferred or routed to review.

#### 4. Domain Object Modeling

`DomainObjectModeler` is a cross-domain role for candidate-only domain object model evolution. It observes synthetic or project-scoped operational signals and proposes missing objects, relation candidates, state candidates, lifecycle delta candidates, permission delta candidates, conflict records, and retention decisions.

Boundary:

- All generated outputs remain `PENDING`.
- Accepted object registries are not mutated.
- Harness workers cannot promote object candidates.
- Every candidate requires evidence lineage.
- Permission-relevant candidates require permission impact notes.
- Patch candidates require schema impact review.
- Private or confidential source material cannot enter shared/global models without an explicit review marker.

#### 5. Seed Pack and Return Pack Discipline

The repository preserves PM seed packs, incoming patch seeds, return manifests, hash inventories, rollback pointers, and validation reports.

Boundary:

- Seed packs are treated as instructions or evidence, not automatic authority.
- Return packs document what was changed and how it was verified.
- Cross-project pointer updates remain candidates until approved.

### What AgentOS CoreSlim Can Do

- Bootstrap a clean AgentOS base for downstream domain projects.
- Run local tests and scripts under explicit bounded authorization.
- Generate candidate-only governance artifacts.
- Preserve rollback and replay evidence.
- Validate kernel boundaries with synthetic tests.
- Package return materials with manifests and hash inventories.
- Separate base-maintenance work from research self-evolution work.

### What AgentOS CoreSlim Must Not Do

- Mutate official theory baselines without explicit authorization.
- Activate production or global registries automatically.
- Promote candidates into accepted state without human or PM gate.
- Use private material to update shared/global models without permission review.
- Treat Harness output as a final governance decision.
- Claim AGI achievement or production readiness from test passage alone.
- Import research-AgentOS outputs into the base without a signed sync seed.

### Repository Layout

```text
agentos_core_slim_v0/   CoreSlim kernel modules and tests
configs/                Local configuration templates
project_baselines/      Project baseline pointers and notes
scripts/                Historical and validation runner scripts
seedpacks/              PM seed packs and patch handoff materials
outputs/                Return packs, validation reports, hash inventories
_incoming/              Received seed packs and integration evidence
```

### Validation

Current local validation:

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```

### Release Boundary

This branch is suitable as an internal base-maintenance candidate. Before a formal public release, review:

- license and distribution policy;
- CI workflow;
- package metadata;
- release tag;
- signed pointer approval;
- public/private data boundary;
- whether `outputs/` should remain in the repository or move to release artifacts.

## 中文

AgentOS CoreSlim 是一个轻量级、以治理边界优先的跨项目 AgentOS 运行时基座。它的目标不是做一个单一聊天机器人、文档解析器、工作流脚本或模型封装，而是为不同领域的 AgentOS 项目提供可复用的内核策略、本地执行边界、候选态演化流程和可审计验证材料。

本分支是干净的 **base-maintenance** 线，刻意与 research-AgentOS self-evolution 输出隔离。研究线输出可以作为证据保存，但不会自动进入基座；只有未来存在 PM 明确批准的同步 seed 时，稳定 patch 才能被导入基座线。

### AgentOS 是什么

AgentOS 是一个面向“有治理的认知工作流”的操作层。它负责协调角色、证据、策略、候选变更、人类审查门和回滚审计，使一个 AgentOS 项目能够在不同领域中安全演化。

在 CoreSlim 基座中，重点是：

- 有边界的本地执行，而不是无约束自动化；
- 由 Kernel 授权，而不是由工具自行决定；
- candidate-only 的演化，而不是直接写入 accepted registry；
- 可 replay 的证据，而不是不可验证的记忆声明；
- rollback 和 audit trail，而不是静默状态变化；
- 跨项目可移植性，而不是一次性项目脚本。

### 当前状态

- 状态：base-maintenance 候选版本，等待 PM / human review
- 分支：`AgentOS`
- 验证：本地测试已通过
- 用途：启动下游 AgentOS 项目，并维护 CoreSlim 基座
- 不声明：生产发布、全局 registry 激活、官方理论基线写入、AGI 达成、自治生产部署

### 推荐使用方式

AgentOS CoreSlim 最适合以“治理基座 + 可替换 runner”的方式使用。runner 是人类或项目维护者操作 AgentOS 任务的交互式编码/执行界面。推荐 runner 包括：

- **Codex**：适合仓库维护、本地代码修改、测试、打包、return pack 和 GitHub 交接。
- **Claude Code**：适合作为另一种本地代码 runner，用于仓库导航、patch 实现和 review 风格工作流。
- **WorkBuddy**：适合偏运营型的 workspace runner，用于任务编排、交接跟踪和日常执行支持。

runner 不是 AgentOS 的最终权威。在 CoreSlim 模型中，runner 应被视为 operator interface。它可以读取指令、编辑文件、运行测试、准备 patch、调用有边界工具，但不能直接提升 candidate、修改 accepted registry，或绕过 Kernel / PM / human review gate。

典型结构是：

```text
Human / PM
  -> Runner: Codex、Claude Code、WorkBuddy 或其他本地操作界面
  -> AgentOS Kernel policy layer
  -> Harness execution layer(s)
  -> Receipts、hash inventories、rollback pointers、return packs
```

### Harness 执行层

AgentOS 可以外接多个 Harness 执行层。Harness 是执行适配器，不是治理权威。不同项目可以根据任务连接不同 Harness：

- 本地 shell / Python test Harness；
- 文档解析或抽取 Harness；
- 数据验证 Harness；
- 浏览器或 UI 自动化 Harness；
- 仿真或 benchmark Harness；
- 打包与发布 Harness；
- 领域专用 Harness，例如 VC、教育、制造、研究、法律、医疗等适配器。

Harness 输出应被视为证据或 execution receipt。它可以支持候选决策，但不会自动成为 accepted truth。是否能够提升，取决于 Kernel policy、evidence lineage、permission review、schema review、retention review，以及 human / PM authorization。

### 快速启动模式

1. 从 `AgentOS` 分支或经过验证的 CoreSlim return pack 开始。
2. 选择 runner，仓库维护通常推荐 Codex，其他本地工作流可选择 Claude Code 或 WorkBuddy。
3. 修改前先读取当前 seed pack、pointer 或 task prompt。
4. 除非有明确授权，否则所有新行为保持 candidate-only。
5. 运行本地测试和任务要求的 Harness 检查。
6. 输出 return pack，包含 manifest、hash inventory、validation report 和 rollback pointer。
7. Pointer update 和 accepted-registry update 在批准前都只能作为 review candidate。

### 核心能力

#### 1. Kernel 有界工具执行

`CodexToolBridge` 提供有边界的本地执行桥。它可以在 Kernel dispatch envelope 明确授权下读取 artifact、运行本地脚本/测试、打包文件、生成 hash inventory、replay receipt，以及回滚本地写入。

边界：

- Tool bridge 是 Harness Plane executor，不是最终决策者。
- 它不能给自己授权。
- 它会阻断外部 API mutation、production deploy、global memory write、global ICM write、git push、legal signature、investment commitment、unbounded web action 等高风险能力。

#### 2. 项目内 ICM 演化

`AutonomousICMEvolutionPolicy` 允许成熟、证据充分的候选项进入项目内 durable artifact，例如 memory unit、operator memory、policy prior、applicability gate 或 quarantine record。

边界：

- 只能写项目内作用域。
- 不写 global memory、production ICM、official theory baseline 或 accepted evidence。
- 必须带 rollback 和 replay 元数据。
- 弱证据保持候选态。
- negative-transfer candidate 进入 quarantine，而不是提升。

#### 3. 基线演化提案协议

`BaselineEvolutionProposalProtocol` 把成熟的项目内学习转化为可供人类审查的 baseline update proposal。

边界：

- 它只提出 S3/S4 或 theory-baseline 更新候选。
- 它不直接应用这些更新。
- 任何 official/global baseline write 仍然需要签署式 human authorization。
- 证据不足、冲突未解决或 negative-transfer 风险高的候选会被延迟或路由到审查。

#### 4. 领域对象建模

`DomainObjectModeler` 是一个跨领域角色，用于 candidate-only 的领域对象模型演化。它从合成或项目内操作信号中发现缺失对象、关系候选、状态候选、生命周期变化候选、权限变化候选、冲突记录和保留决策。

边界：

- 所有输出保持 `PENDING`。
- 不写 accepted object registry。
- Harness worker 不能提升 object candidate。
- 每个 candidate 都必须有 evidence lineage。
- 权限相关候选必须有 permission impact notes。
- Patch candidate 必须经过 schema impact review。
- private/confidential 材料不能在没有显式审查 marker 的情况下进入 shared/global model。

#### 5. Seed Pack 与 Return Pack 纪律

仓库保存 PM seed pack、incoming patch seed、return manifest、hash inventory、rollback pointer 和 validation report。

边界：

- Seed pack 是指令或证据，不是自动授权。
- Return pack 记录改了什么、如何验证。
- Cross-project pointer update 在批准前只能是 candidate。

### AgentOS CoreSlim 能做什么

- 为下游领域项目启动一个干净的 AgentOS 基座。
- 在显式授权下运行本地测试和脚本。
- 生成 candidate-only 的治理 artifact。
- 保存 rollback 和 replay 证据。
- 用合成测试验证内核边界。
- 用 manifest 和 hash inventory 打包回传材料。
- 将 base-maintenance 工作与 research self-evolution 工作分离。

### AgentOS CoreSlim 不能做什么

- 未经明确授权写入官方理论基线。
- 自动激活生产或全局 registry。
- 绕过 human / PM gate 把候选提升为 accepted。
- 未经权限审查，用 private material 更新 shared/global model。
- 把 Harness 输出当作最终治理决策。
- 因测试通过就宣称 AGI 达成或生产就绪。
- 未经签署式 sync seed，把 research-AgentOS 输出导入基座。

### 仓库结构

```text
agentos_core_slim_v0/   CoreSlim 内核模块与测试
configs/                本地配置模板
project_baselines/      项目基线指针与说明
scripts/                历史 runner 与验证脚本
seedpacks/              PM seed pack 与 patch 交接材料
outputs/                return pack、验证报告、hash inventory
_incoming/              已接收的 seed pack 与集成证据
```

### 验证结果

当前本地验证：

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```

### 发布边界

本分支适合作为内部基座维护候选版本。正式公开 release 前，建议复核：

- license 与分发策略；
- CI workflow；
- package 元数据；
- release tag；
- 经签署/确认的 pointer approval；
- public/private 数据边界；
- `outputs/` 是否应保留在仓库中，或改为 release artifact。
