# 依赖与许可证说明

RouteTune 自身采用 [Apache License 2.0](../LICENSE)。项目只提交自身源代码、前端静态资源和用于演示的非敏感样例，不打包第三方模型权重或供应商数据。

## 直接依赖

下表用于帮助部署者快速审阅直接依赖；具体版本由 `uv.lock` 锁定，许可证以各依赖发布包中的原始文件为准。

| 范围 | 依赖 | 上游许可证 |
|---|---|---|
| Runtime | FastAPI | MIT |
| Runtime | joblib | BSD-3-Clause |
| Runtime | NumPy | BSD-3-Clause |
| Runtime | scikit-learn | BSD-3-Clause |
| Runtime | Uvicorn | BSD-3-Clause |
| Optional benchmark | Apache Arrow / PyArrow | Apache-2.0 |
| Optional NLP | spaCy | MIT |
| Build | Hatchling | MIT |
| Development | HTTPX | BSD-3-Clause |
| Development | pytest | MIT |
| Development | Ruff | MIT |

RouteTune 通过 OpenAI-compatible HTTP API 调用外部 Provider 或私有模型服务。API 兼容并不授予模型权重、输出或服务条款之外的权利；部署者仍需分别遵守所选 Provider、模型和数据集的许可证及使用条款。

## 维护流程

新增或升级直接依赖时，Pull Request 应：

1. 说明引入理由和是否存在更小的替代方案；
2. 更新 `pyproject.toml` 与锁文件；
3. 核对上游许可证、安全公告和分发限制；
4. 更新本页，并在必要时保留上游 NOTICE 或署名文件；
5. 不提交来源不明的模型、数据集、字体、图标或客户材料。

锁文件包含传递依赖的精确版本，是生成完整软件物料清单与许可证报告的依据。生产部署方应结合自身发布形式和合规要求执行最终审查。
