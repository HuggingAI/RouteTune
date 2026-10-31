# RouteTune 本地演示与验收指南

本文帮助使用者验证完整产品闭环，也供维护者在发布前执行同一套可重复验收。默认只使用本地 Mock Provider，不需要 API Key，不会产生模型费用或发送业务数据。

## 1. 启动

推荐使用 Docker Compose：

```bash
cp .env.example .env
docker compose up --build -d
curl http://127.0.0.1:8787/health
```

健康响应必须满足：

- `status` 为 `ok`；
- `provider_mode` 为 `demo`；
- `mock-fast`、`mock-balanced`、`mock-strong` 都在 `online_models` 中；
- `routing_targets` 分别包含 `tier_1`、`tier_3`、`tier_5`。

## 2. 自动发布验收

在服务启动后执行：

```bash
uv run python scripts/smoke_workflow.py
```

脚本会在运行中的实例里完成：

1. 冷启动统一 API 调用；
2. 导入固定的 12 条公开 Demo 数据；
3. 三个本地候选模型共 36 次运行；
4. 写入明确标记为 synthetic 的机械验收标注；
5. 训练 TF-IDF + Logistic Regression Router；
6. 创建并发布不可变 Release；
7. 发布后再次调用统一 API；
8. 验证 Trace、Token、延迟与失败统计。

结果写入 `artifacts/smoke/workflow-report.json`。`artifacts/` 已被 Git 忽略。

> 自动标注只用于证明系统闭环可以执行，不代表模型质量，也不得用于宣称准确率、
> 成本节省或客户效果。比赛录屏中的 Pass/Fail 应由演示者在界面中实际完成。

## 3. 人工录屏路径

使用 `examples/enterprise-pilot/enterprise-pilot-smoke-12.jsonl`，按以下顺序录制：

1. 在“API 接入”发送一条请求，展示统一入口和 Trace ID；
2. 在“模型池”展示三个路由层级及可用状态；
3. 在“路由数据”选择模型对比模式，导入 12 条数据并运行；
4. 在“人工标注”查看 Ground Truth，对输出执行 Pass/Fail；
5. 训练推荐的 Auto Router，展示实际解析为 TF-IDF；
6. 在“Router 版本”发布候选版本；
7. 再次调用统一 API，展示新的 Release ID；
8. 在“请求记录”与“运行大盘”展示路由、模型、延迟、Input/Output Token 和失败统计。

人工录屏使用一套新的空数据库，避免自动验收产生的记录干扰画面。

## 4. 清理

```bash
docker compose down
```

只有明确不再需要本地演示数据时才使用 `docker compose down -v`。
