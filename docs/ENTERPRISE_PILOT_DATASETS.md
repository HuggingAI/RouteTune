# 企业场景公开数据试跑

> 目标不是证明产品价值，而是用不同难度的真实公开任务完整走通“上传 → 多模型执行 → 人工标注 → 训练 → 候选版本 → 发布 → 统一 API → Trace”闭环。

## 1. 数据组合

| 数据集 | 企业场景 | 任务 | 难度 | 本试验数量 | 许可 |
|---|---|---|---|---:|---|
| [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) | 银行客服 | 77 类意图分类 | 易 | 10 | CC BY 4.0 |
| [MASSIVE](https://github.com/alexa/massive) | 中文企业助手 | 60 类意图分类 | 中 | 10 | CC BY 4.0 |
| [FinQA](https://github.com/czyssrs/FinQA) | 财报分析 | 基于证据的数值推理 | 难 | 10 | MIT |

这三组任务不是为了模拟一家具体客户，而是故意形成清晰梯度：便宜模型应能覆盖大量标签分类，复杂财务计算更可能需要高层模型。这样才能验证 RouteTune 是否学会把请求送到“最低但足够”的模型层级。

CUAD 的合同条款识别也适合企业场景，但单份合同输入更长、人工判断成本更高，留到第二轮扩展，不加入本次 30 条试跑。

## 2. 文件与格式

可直接上传：

- [`../examples/enterprise-pilot/enterprise-pilot-smoke-12.jsonl`](../examples/enterprise-pilot/enterprise-pilot-smoke-12.jsonl)
- [`../examples/enterprise-pilot/enterprise-pilot-full-30.jsonl`](../examples/enterprise-pilot/enterprise-pilot-full-30.jsonl)

每行都有：

```json
{
  "id": "stable-source-id",
  "messages": [
    {"role": "system", "content": "task instruction"},
    {"role": "user", "content": "public test input"}
  ],
  "metadata": {
    "source": "dataset",
    "task_family": "task type",
    "difficulty": "easy | medium | hard",
    "ground_truth_type": "exact_label | final_value",
    "ground_truth": "review-only answer"
  },
  "max_tokens": 512,
  "temperature": 0
}
```

Ground Truth 只位于 metadata。RouteTune 调用候选模型时只发送 messages；“人工标注”页会在候选输出上方直接展示 GT 和判定类型。没有 GT 的真实业务样本仍可上传，由用户按业务标准人工判断。

## 3. 推荐完整流程

1. 在“路由数据”导入 JSONL 并选择候选模型。
2. 直接勾选 3 个真实模型，建议分别覆盖 `tier_1`、`tier_2`、`tier_3`。
3. 上传 12 条 smoke。页面默认使用 `max_tokens=512`、`temperature=0`；页面参数覆盖 JSONL，留空时依次使用 JSONL 单条值、模型池默认值和 Provider 默认值。
4. 执行批次。12 条 × 3 个模型 = 36 次串行请求，适合免费 API 的低并发限制。
5. 展开每条参考答案，对各模型输出做 Pass/Fail 标注。意图分类要求标签精确匹配；FinQA 以最终数值为准，允许表述差异。
6. 完整标注后训练候选 Bundle，在“Router 版本”页确认 Router、模型池和安全快照，再人工发布。
7. 用 Playground 调用统一 `/v1/chat/completions`，查看 Release、路由标签、实际模型和 Trace。
8. smoke 无工程错误后，再上传 30 条完整集；30 条 × 3 个模型 = 90 次调用。

候选模型由本次批次直接保存，不需要修改模型的能力标签。执行前可由“样本数 × 候选模型数”确认请求总量。

## 4. 可复现生成

本仓库保存的是小规模转换结果。重新生成时下载以下原始文件：

- Banking77 官方 `banking77/test.csv`；
- MASSIVE Hugging Face 官方镜像 `zh-CN/massive-test.parquet`，本次使用 revision `0bd9da89cf338b4481baf431430431e56b460853`；
- FinQA 官方 `dataset/test.json`，本次使用 revision `0f16e2867befa6840783e58be38c9efb9229d742`。

然后运行：

```bash
python scripts/prepare_enterprise_pilot.py \
  --banking /path/to/banking77-test.csv \
  --massive /path/to/massive-test.parquet \
  --finqa /path/to/finqa-test.json \
  --output-dir examples/enterprise-pilot
```

读取 Parquet 的准备脚本需要项目的 `benchmark` 可选依赖（`pyarrow`）；RouteTune 服务运行时不依赖它。输出 `manifest.json` 保存输入文件 SHA-256，便于确认来源是否相同。
