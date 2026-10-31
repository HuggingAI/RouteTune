# Contributing to RouteTune

感谢你帮助 RouteTune 变得更好。项目当前坚持一个边界：功能必须直接服务于“真实请求 → 模型输出 → 人工反馈 → Router 训练 → 受控发布 → 线上回流”这条闭环。

提交贡献即表示你同意遵守 [社区行为准则](CODE_OF_CONDUCT.md)。较大的方向请先阅读 [路线图](ROADMAP.md) 与 [项目治理](GOVERNANCE.md)，避免实现与产品边界冲突的功能。

## 开始开发

```bash
uv sync
uv run pytest
uv run ruff check .
```

提交代码前请：

1. 先搜索已有 Issue；较大的产品方向先开 Discussion 或 Issue 对齐范围；
2. 为行为变化补充测试；
3. 不提交 API Key、客户数据、数据库、训练产物或本机配置；
4. 保持 PR 小而聚焦，并说明用户可见变化与验证方式；
5. 对数据格式、Release 或存储结构的变化说明兼容性影响。

## Issue 与 Pull Request

- Bug 请包含版本、部署方式、最小复现、预期行为和已经脱敏的日志；
- 功能建议请先说明真实场景、现有替代方式和可验证的成功标准；
- 不要在公开内容中粘贴 API Key、客户输入、内部域名或未脱敏 Trace；
- Pull Request 请关联 Issue，说明改动、验证结果、兼容性及文档影响；
- 影响 API、Bundle、Release 或数据库的改动必须提供迁移与回滚说明。

维护者会优先处理可复现 Bug、安全问题、破坏数据兼容性的风险，以及能够改善真实路由闭环的小而明确的改进。

提交贡献即表示你同意以项目的 Apache-2.0 License 发布该贡献。
