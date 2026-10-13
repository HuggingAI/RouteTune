# Contributing to RouteTune

感谢你帮助 RouteTune 变得更好。项目当前坚持一个边界：功能必须直接服务于“真实请求 → 模型输出 → 人工反馈 → Router 训练 → 受控发布 → 线上回流”这条闭环。

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

提交贡献即表示你同意以项目的 Apache-2.0 License 发布该贡献。
