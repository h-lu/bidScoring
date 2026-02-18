# Question Bank Design (Archived Plan)

## 1. 背景

该计划用于定义早期问题集设计方向，现已并入 policy pack 体系。

## 2. 现状（已落地）

- 问题集由 `pack + overlay` 管理
- 生产默认为 `cn_medical_v1 + strict_traceability`
- 运行时在输出中回显：
  - `observability.question_bank.pack_id`
  - `observability.question_bank.overlay`
  - `observability.question_bank.question_count`

## 3. 对应实现

- `config/policy/packs/cn_medical_v1/*`
- `bid_scoring/pipeline/application/question_context.py`
- `bid_scoring/pipeline/interfaces/cli.py`

## 4. 归档说明

本文件仅保留历史设计意图，当前以 `config/policy/packs/*` 为准。
