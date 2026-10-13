# RouteTune Engine 数据契约

MVP 使用一行一个请求的 JSONL。每条请求必须包含相同的候选模型集合，每个候选结果都要绑定不可变 Annotation。

```json
{
  "request_id": "req-001",
  "request_text": "请总结这份故障报告",
  "metadata": {
    "scenario": "summary",
    "baseline_model_id": "balanced"
  },
  "candidates": [
    {
      "model_id": "economy",
      "annotation": {
        "passed": false,
        "failure_reason": "MISSING_KEY_FACT",
        "score": 2,
        "schema_version": "pass-fail/v1"
      },
      "cost_rank": 1,
      "latency_ms": 220,
      "output_hash": "sha256-of-immutable-output"
    },
    {
      "model_id": "premium",
      "annotation": {
        "passed": true,
        "score": 5,
        "schema_version": "pass-fail/v1"
      },
      "cost_rank": 2,
      "latency_ms": 910,
      "output_hash": "sha256-of-immutable-output"
    }
  ]
}
```

当前实现刻意只使用以下训练信号：

- `request_text`；
- `candidate.model_id`；
- `annotation.passed`。

成本等级、延迟和失败原因用于决策或报告，不作为第一版分类器的训练捷径。`cost_rank` 是候选模型之间稳定的相对顺序：`1` 最便宜，数字越大越贵；它不是金额。数据按 `request_id` 切分为训练、校准和测试集，同一个请求的多个候选结果不会跨集合泄漏。

## Bundle

训练产生的 Bundle 包含：

- 每个候选模型的字符级 TF-IDF + Logistic Regression 分类器；
- 每个候选模型的 Platt 风格概率校准器；
- 词表覆盖率 OOD 阈值；
- 质量阈值；
- 候选模型成本等级和平均延迟；
- 数据集指纹和 Bundle 版本。

## Decision

```json
{
  "request_id": "req-new",
  "selected_model_id": "balanced",
  "used_fallback": false,
  "reason": "LOWEST_COST_ABOVE_QUALITY_THRESHOLD",
  "prediction": {
    "bundle_version": "bundle-...",
    "abstained": false,
    "candidates": []
  }
}
```

没有候选模型达到质量门槛，或请求被判定为 OOD 时，MVP 回退到训练集与校准集上实际通过率最高的候选模型；通过率并列时优先选择成本等级更高者。绝不把“最贵”未经验证地等同于“最强”。
