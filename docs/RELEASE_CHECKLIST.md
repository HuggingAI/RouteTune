# 维护者发布清单

> 本文件是维护者作业记录，不是最终用户使用文档。首次开源准备已经完成；后续版本应从当前主分支重新执行适用的验收项。

## 已完成

- [x] README 首屏定位、能力边界和快速开始
- [x] Apache-2.0 License
- [x] Dockerfile、Compose、健康检查和持久化 volumes
- [x] 零 Key 本地 Mock 完整调用链路
- [x] 环境变量与单一 `.env` Key 配置方式
- [x] 部署、安全与贡献文档
- [x] GitHub Actions：Ruff、测试与 Docker build
- [x] 中英文发布文案和 Demo 顺序
- [x] wheel/sdist 构建与静态资源打包检查
- [x] 密钥、本机路径和本地配置提交检查

## 每次发布前人工确认

- [ ] 检查提交中没有 `.env`、本地 TOML、数据库、客户数据和训练 artifacts
- [ ] 在正常 Docker Hub 网络下完成一次 `docker compose up --build`
- [ ] 用公开样例数据截取运行大盘、人工标注、Router 版本三张图
- [ ] 确认 GitHub Description、Topics、Social preview 和文档链接仍然准确
- [ ] 确认 GitHub Discussions 与 Private vulnerability reporting 可用
- [ ] 推送 `main`，确认 CI 全绿并完成依赖许可证审阅
- [ ] 创建对应版本的 tag 和 GitHub Release，附迁移与回滚说明
- [ ] 发布版本说明，并把真实反馈收敛到 Issues/Discussions

## `v0.1.0` Release 标题

```text
RouteTune v0.1.0 — Train and publish an LLM router from real feedback
```
## `v0.1.0` Release 摘要

```text
The first open-source RouteTune release delivers a complete single-node routing loop:
model pool, redaction, JSONL batch runs, focused pass/fail review, probabilistic router
training, immutable releases, an OpenAI-compatible API, traces, and operational metrics.

Docker Compose includes local mock models, so the entire workflow can be evaluated without
an external provider key. This release is an early single-node MVP and does not yet claim
multi-replica or high-availability deployment support.
```
