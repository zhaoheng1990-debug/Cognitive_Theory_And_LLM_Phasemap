# AgentOS CoreSlim Base / AgentOS CoreSlim 基座

## English

AgentOS is a project operating layer for AI-assisted work that needs memory, evidence, review, rollback, and repeatable execution. It is not trying to replace Codex, Claude Code, WorkBuddy, or any other runner. It gives those runners a governed place to work.

The CoreSlim base is the small, clean starting point for that operating layer. It is meant for people who want to build domain AgentOS projects without carrying over unstable research experiments, private context leaks, or one-off automation habits.

### The Short Version

AgentOS helps a project answer a few practical questions every time an AI runner does work:

- What is the runner allowed to do?
- What evidence did it use?
- What changed, and can we replay or roll it back?
- Is this only a candidate, or has a human/PM approved it?
- Which Harness checked the result?
- Can this lesson be reused in another project without importing the wrong context?

If ordinary agent tooling is the hand that edits, runs, searches, or packages, AgentOS is the layer that keeps the work legible and governable.

### Why This Exists

AI coding and workflow tools are already useful, but long-running projects quickly develop a different problem: the work becomes hard to trust.

Files change. Prompts drift. Local scripts produce outputs. A runner says something has been verified, but the evidence is not packaged. A project learns something useful, but nobody knows whether it belongs only to this project or should become a reusable rule. A research branch produces ideas, but the base system should not quietly absorb them.

AgentOS CoreSlim exists to make this kind of work calmer. It keeps useful AI execution, but adds a simple discipline around it:

- keep candidates separate from accepted state;
- keep evidence next to the decision;
- keep runners replaceable;
- keep Harnesses as execution layers, not authorities;
- keep rollback pointers and return packs;
- keep research evolution separate from the clean base.

### The Role AgentOS Plays

AgentOS is best understood as an operating layer around cognitive work. It does not have to be the model, the editor, the browser, the test runner, or the deployment tool. Instead, it coordinates them.

In a typical project:

- a human or PM gives the goal;
- a runner such as Codex, Claude Code, or WorkBuddy performs the local work;
- AgentOS supplies the policy, role boundaries, candidate rules, and review gates;
- one or more Harnesses execute checks or domain-specific operations;
- the result is returned with receipts, hashes, manifests, validation notes, and rollback information.

This makes AgentOS useful when a project needs more than a clever assistant. It is for work where the path matters, not only the final answer.

### What Problems It Helps Solve

**Scattered AI work.** AgentOS turns isolated runner sessions into traceable project cycles with seed inputs, outputs, validation, and return packs.

**Tool overreach.** Runners and Harnesses can do useful work, but they do not get to promote candidates or rewrite accepted registries by themselves.

**Lost context.** Project learning can be stored as project-scoped candidate memory or policy prior, with enough evidence to review later.

**Unsafe reuse.** A lesson from one domain does not automatically become a global rule. AgentOS keeps cross-project transfer explicit.

**Research/base confusion.** Experimental self-evolution outputs can be preserved as evidence without silently entering the clean base.

**Hard rollback.** Changes are easier to inspect and unwind when manifests, hash inventories, receipts, and rollback pointers are produced as part of the workflow.

### How To Use It Without Making Life Miserable

Start small. You do not need a large agent platform before AgentOS becomes useful.

1. Pick a runner you are comfortable using.
2. Give it a seed, pointer, issue, or task prompt.
3. Let it work inside the AgentOS rules: candidate-first, evidence-backed, reviewable.
4. Attach only the Harnesses needed for this task.
5. Run the checks.
6. Ask for a return pack: what changed, what was verified, what remains pending, and how to roll back.
7. Promote only what a human/PM has approved.

The pleasant path is: use your favorite runner for flow, use AgentOS for memory and governance, use Harnesses for execution evidence.

### Recommended Runners

AgentOS CoreSlim works well with replaceable runners. The runner is the interactive surface; AgentOS is the governance layer.

- **Codex** is a good default for repository maintenance, local code edits, tests, packaging, return packs, and GitHub handoff.
- **Claude Code** is useful as an alternate local coding runner for navigation, patching, and review-style workflows.
- **WorkBuddy** is useful when the project needs workspace coordination, task tracking, handoff continuity, or day-to-day operating support.
- Other runners can be used if they can follow the same rules: read the seed, work locally, respect candidate boundaries, run checks, and return evidence.

The runner should not be treated as the final authority. It may edit, test, package, inspect, and propose. It should not directly promote candidates, mutate accepted registries, bypass review gates, or turn private project material into shared/global state.

### Harness Execution Layers

A Harness is an execution adapter. It gives AgentOS a way to run or verify something in the outside world, but it does not own the governance decision.

Useful Harness layers may include:

- local shell, Python, pytest, or build Harnesses;
- document parsing and extraction Harnesses;
- data validation Harnesses;
- browser, UI, or app automation Harnesses;
- simulation, benchmark, or evaluation Harnesses;
- packaging and release Harnesses;
- domain Harnesses for VC, education, manufacturing, research, legal, healthcare, or other project worlds.

Harness output is evidence. It can support a decision, but it does not become accepted truth on its own. AgentOS keeps the distinction between "this check ran" and "this change is approved."

### A Normal Work Cycle

```text
Human / PM
  -> Runner: Codex, Claude Code, WorkBuddy, or another local operator
  -> AgentOS CoreSlim policy and role boundaries
  -> Harness execution layer(s)
  -> Evidence, receipts, validation, hashes, rollback pointer
  -> Human / PM review
  -> Candidate stays pending, gets revised, or is approved
```

### What Is Inside CoreSlim

CoreSlim currently provides a compact set of base mechanisms:

- bounded local tool execution through `CodexToolBridge`;
- project-scoped ICM evolution policy for evidence-backed candidates;
- baseline evolution proposal protocol for human-reviewable updates;
- `DomainObjectModeler` for candidate-only domain object modeling;
- seed pack and return pack discipline;
- tests that check the main governance boundaries.

These are intentionally small. CoreSlim is a base, not a finished product suite.

### What AgentOS CoreSlim Can Do

- Bootstrap a clean AgentOS base for downstream projects.
- Help a runner operate under explicit boundaries.
- Keep new behavior candidate-only until reviewed.
- Preserve evidence, validation notes, hash inventories, and rollback pointers.
- Support project-scoped learning without mutating global truth.
- Package handoff materials so another window, runner, or maintainer can continue.
- Separate base maintenance from research self-evolution.

### What It Should Not Be Used For

- Autonomous production deployment.
- Legal, financial, medical, or operational commitments without human authority.
- Silent mutation of accepted registries or official baselines.
- Turning Harness output into final truth without review.
- Importing private material into shared/global models without permission review.
- Claiming AGI or production readiness because local tests pass.

### Current Status

- Branch: `AgentOS`
- Status: base-maintenance candidate for PM/human review
- Intended use: bootstrap and maintain AgentOS CoreSlim based projects
- Validation:

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```

Before a formal public release, review license/distribution policy, CI workflow, package metadata, release tags, signed pointer approval, and public/private data boundaries.

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

## 中文

AgentOS 是一个给 AI 协作项目使用的操作层。它关心的不是“换一个聊天机器人”，而是让一个长期项目在使用 Codex、Claude Code、WorkBuddy 或其他 runner 时，仍然能保留记忆、证据、审查、回滚和可复现的执行过程。

CoreSlim 是这个操作层的干净基座。它适合用来启动下游 AgentOS 项目，也适合维护跨项目共用的基础规则。它刻意不把不稳定的 research self-evolution 输出、私人上下文和一次性自动化习惯直接带进基座。

### 一句话说明

每当一个 AI runner 替项目做事时，AgentOS 帮项目回答这些问题：

- 这个 runner 被允许做什么？
- 它用了什么证据？
- 它改了什么，能不能 replay 或 rollback？
- 这是候选结果，还是已经被 human/PM 批准？
- 哪个 Harness 检查了结果？
- 这个经验能否跨项目复用，还是只能留在当前项目里？

如果普通 agent 工具负责编辑、运行、搜索、打包，那么 AgentOS 负责让这些工作变得可追踪、可审查、可交接。

### 为什么需要 AgentOS

AI 编码和工作流工具已经很好用，但项目一旦持续变长，问题就会变成：工作越来越难被信任。

文件改了，prompt 变了，本地脚本生成了输出。runner 说已经验证过，但证据没有打包。项目学到了一条经验，却不知道它只适用于本项目，还是可以变成跨项目规则。研究线产生了有价值的想法，但干净基座不应该悄悄吸收它们。

AgentOS CoreSlim 想解决的正是这种混乱。它保留 AI 执行带来的速度，但给它加上一套轻量纪律：

- candidate 和 accepted 分开；
- 证据和决策放在一起；
- runner 可以替换；
- Harness 只是执行层，不是治理权威；
- return pack、hash inventory、rollback pointer 成为工作流的一部分；
- research evolution 和 clean base 分开维护。

### AgentOS 扮演什么角色

AgentOS 可以理解为认知工作流外面的一层“项目操作系统”。它不一定亲自做模型、编辑器、浏览器、测试器或部署器。它的作用是协调这些东西。

一个典型项目里：

- human 或 PM 给出目标；
- Codex、Claude Code、WorkBuddy 等 runner 完成本地操作；
- AgentOS 提供策略、角色边界、候选态规则和审查门；
- 一个或多个 Harness 执行检查或领域动作；
- 最终输出 receipts、hash、manifest、validation notes 和 rollback 信息。

所以 AgentOS 适合的场景，不只是“找一个聪明助手回答问题”，而是那些过程本身也很重要的工作。

### 它能解决哪些实际问题

**AI 工作分散。** AgentOS 把零散 runner 会话整理成带 seed、输出、验证和 return pack 的项目周期。

**工具越权。** runner 和 Harness 可以做事，但不能自己把候选结果提升为 accepted，也不能自己改 accepted registry。

**上下文丢失。** 项目学到的东西可以进入项目内候选记忆或 policy prior，并保留足够证据供以后审查。

**复用不安全。** 一个领域里的经验不会自动变成全局规则。跨项目迁移必须显式发生。

**研究线和基座混在一起。** research self-evolution 输出可以作为证据保存，但不会静默进入 clean base。

**回滚困难。** 当 manifest、hash inventory、receipt 和 rollback pointer 成为固定输出时，后续检查和撤回会轻松很多。

### 怎样愉快地用起来

先从小任务开始，不需要一上来搭一个庞大的 agent 平台。

1. 选择一个你顺手的 runner。
2. 给它 seed、pointer、issue 或 task prompt。
3. 让它按 AgentOS 规则工作：先候选、带证据、可审查。
4. 只接入当前任务真正需要的 Harness。
5. 运行检查。
6. 要求输出 return pack：改了什么、验证了什么、什么仍然 pending、如何回滚。
7. 只有 human/PM 批准的内容才能被提升。

最舒服的使用方式是：用你喜欢的 runner 保持工作流顺畅，用 AgentOS 管记忆和治理，用 Harness 提供执行证据。

### 推荐 Runner

AgentOS CoreSlim 适合搭配可替换 runner 使用。runner 是交互界面，AgentOS 是治理层。

- **Codex**：适合仓库维护、本地代码修改、测试、打包、return pack 和 GitHub 交接。
- **Claude Code**：适合作为另一种本地编码 runner，用于仓库导航、patch 实现和 review 风格工作流。
- **WorkBuddy**：适合需要任务编排、交接连续性、日常执行支持的 workspace。
- 其他 runner 也可以使用，只要它能遵守同一套规则：读取 seed，本地执行，尊重候选边界，运行检查，返回证据。

runner 不应该被当作最终权威。它可以编辑、测试、打包、检查和提出建议；但不应该直接提升 candidate、修改 accepted registry、绕过审查门，或把私人项目材料写入 shared/global state。

### Harness 执行层

Harness 是执行适配器。它让 AgentOS 能在外部世界运行或验证某件事，但它不拥有治理决策权。

可接入的 Harness 包括：

- 本地 shell、Python、pytest 或 build Harness；
- 文档解析与抽取 Harness；
- 数据验证 Harness；
- 浏览器、UI 或 app 自动化 Harness；
- 仿真、benchmark 或 evaluation Harness；
- 打包与发布 Harness；
- 面向 VC、教育、制造、研究、法律、医疗等领域的专用 Harness。

Harness 输出是证据。它可以支持决策，但不会自动成为 accepted truth。AgentOS 保留“检查已经运行”和“变更已经批准”之间的区别。

### 一个正常工作周期

```text
Human / PM
  -> Runner: Codex、Claude Code、WorkBuddy 或其他本地操作界面
  -> AgentOS CoreSlim policy 与 role boundary
  -> Harness execution layer(s)
  -> Evidence、receipts、validation、hashes、rollback pointer
  -> Human / PM review
  -> Candidate 保持 pending、继续修改，或被批准
```

### CoreSlim 里面有什么

CoreSlim 当前提供一组小而干净的基座机制：

- 通过 `CodexToolBridge` 进行有边界的本地工具执行；
- 项目内 ICM 演化策略，用于有证据支撑的候选项；
- baseline evolution proposal protocol，用于生成可由人类审查的基线更新提案；
- `DomainObjectModeler`，用于 candidate-only 的领域对象建模；
- seed pack 和 return pack 纪律；
- 覆盖主要治理边界的测试。

这些能力刻意保持小。CoreSlim 是基座，不是一个已经完成的产品套件。

### AgentOS CoreSlim 能做什么

- 为下游项目启动一个干净的 AgentOS 基座。
- 帮 runner 在明确边界内工作。
- 让新行为在审查前保持 candidate-only。
- 保存证据、验证说明、hash inventory 和 rollback pointer。
- 支持项目内学习，但不直接改写全局真值。
- 打包交接材料，让另一个窗口、runner 或维护者可以继续。
- 将 base maintenance 与 research self-evolution 分开。

### 不应该用它做什么

- 自治生产部署。
- 在没有人类授权时做法律、金融、医疗或运营承诺。
- 静默修改 accepted registry 或官方 baseline。
- 把 Harness 输出直接当作最终真值。
- 未经权限审查，把私人材料写入 shared/global model。
- 因为本地测试通过就宣称 AGI 达成或生产就绪。

### 当前状态

- 分支：`AgentOS`
- 状态：base-maintenance 候选版本，等待 PM/human review
- 用途：启动和维护基于 AgentOS CoreSlim 的项目
- 验证：

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```

正式公开 release 前，建议复核 license 与分发策略、CI workflow、package 元数据、release tag、签署式 pointer approval，以及 public/private 数据边界。

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
