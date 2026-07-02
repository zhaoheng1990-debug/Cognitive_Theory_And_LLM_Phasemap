# AgentOS CoreSlim Base / AgentOS CoreSlim 基座

## English

AgentOS CoreSlim is a lightweight, governance-first runtime base for cross-project AgentOS work. This branch preserves a clean base-maintenance line separated from research self-evolution outputs.

This repository currently packages the CoreSlim kernel primitives, tests, configuration templates, seed packs, project baselines, and local validation artifacts needed to bootstrap downstream AgentOS projects.

### Current Status

- Status: base-maintenance candidate for review
- Branch: `AgentOS`
- Scope: CoreSlim infrastructure, bounded kernel policies, candidate-only evolution flows, tests, and handoff materials
- Not included: production deployment, global registry activation, official theory-baseline mutation, or AGI achievement claims

### Core Capabilities

- Bounded Codex tool bridge for local execution under kernel authorization
- Project-scoped autonomous ICM evolution policy with rollback/replay metadata
- Baseline evolution proposal protocol for human-authorized promotion review
- DomainObjectModeler role for candidate-only domain object model evolution
- Synthetic boundary tests for permission, schema, registry, promotion, and evidence constraints

### Repository Layout

```text
agentos_core_slim_v0/   CoreSlim kernel code and tests
configs/                Local configuration templates
project_baselines/      Project baseline pointers and notes
scripts/                Historical and validation runner scripts
seedpacks/              PM seed packs and patch handoff materials
outputs/                Return packs, validation reports, and release-size checks
_incoming/              Received seed packs awaiting or documenting integration
```

### Validation

The current local validation passed:

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```

### Release Boundary

This branch is suitable as an internal base-maintenance candidate. Before a formal public release, add or review:

- license policy
- CI workflow
- package metadata
- release tags
- signed pointer approval

## 中文

AgentOS CoreSlim 是一个轻量级、以治理边界优先的跨项目 AgentOS 运行时基座。本分支保存的是干净的 base-maintenance 线，与 research-AgentOS self-evolution 输出明确隔离。

当前仓库打包了 CoreSlim 内核原语、测试、配置模板、seed pack、项目基线和本地验证材料，可用于下游 AgentOS 项目的启动与基座复用。

### 当前状态

- 状态：基座维护候选版本，等待 PM / human review
- 分支：`AgentOS`
- 范围：CoreSlim 基础设施、有界内核策略、candidate-only 演化流、测试与交接材料
- 不包含：生产部署、全局 registry 激活、官方理论基线写入、AGI 达成声明

### 核心能力

- 在 AgentOSKernel 授权下运行的本地有界 Codex tool bridge
- 带 rollback / replay 元数据的项目内 ICM 演化策略
- 面向人工授权提升的 baseline evolution proposal protocol
- `DomainObjectModeler` 角色，用于 candidate-only 的领域对象模型演化
- 覆盖权限、schema、registry、promotion、evidence 边界的合成测试

### 仓库结构

```text
agentos_core_slim_v0/   CoreSlim 内核代码与测试
configs/                本地配置模板
project_baselines/      项目基线指针与说明
scripts/                历史 runner 与验证脚本
seedpacks/              PM seed pack 与 patch 交接材料
outputs/                return pack、验证报告和体量检查
_incoming/              已接收的 seed pack 与集成记录
```

### 验证结果

当前本地验证已通过：

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```

### 发布边界

本分支适合作为内部基座维护候选版本。正式公开 release 前，建议补齐或复核：

- license 策略
- CI workflow
- Python package 元数据
- release tag
- 经签署/确认的 pointer approval
