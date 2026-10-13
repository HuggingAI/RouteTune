# Security Policy

## Supported version

RouteTune 尚处于早期开源阶段，安全修复仅针对默认分支的最新版本。

## Reporting a vulnerability

请不要为安全漏洞创建公开 Issue，也不要附带真实 API Key、客户输入或未脱敏 Trace。请通过 GitHub 仓库的 **Security → Report a vulnerability** 私下提交：

<https://github.com/HuggingAI/RouteTune/security/advisories/new>

报告中请包含受影响版本、复现步骤、影响和建议修复方式。维护者会尽快确认，并在修复发布后协调披露。

## Deployment notes

- 生产环境必须设置 `ROUTETUNE_API_KEY` 并使用 HTTPS；
- 默认管理台与业务 API 共用一个 Bearer Token，不应直接暴露在公网；
- Provider Key 应通过环境变量或秘密管理系统注入；
- Trace 只应保存脱敏正文，敏感场景可启用 `metadata_only`；
- 当前 SQLite 单机版本不提供租户隔离或细粒度 RBAC。
