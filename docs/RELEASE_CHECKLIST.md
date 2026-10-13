# 首次开源发布清单

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

## 发布前人工确认

- [ ] 将本地仓库连接到 `git@github.com:HuggingAI/RouteTune.git`
- [ ] 检查初始 commit 中没有 `.env`、本地 TOML、数据库和 artifacts
- [ ] 在正常 Docker Hub 网络下完成一次 `docker compose up --build`
- [ ] 用公开样例数据截取大盘、统一审阅、版本发布三张图
- [ ] 在 GitHub 设置 Description、Topics 和 Social preview
- [ ] 启用 GitHub Discussions 与 Private vulnerability reporting
- [ ] 推送 `main`，确认 CI 全绿
- [ ] 创建 `v0.1.0` tag 和 GitHub Release
- [ ] 发布首发帖，并把真实反馈收敛到 Issues/Discussions

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
