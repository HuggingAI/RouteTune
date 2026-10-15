# RouteTune 部署指南

本文对应当前单实例开源版本。默认 Docker Compose 配置面向本机体验和小团队自托管，不是多副本高可用方案。

## 最快启动

```bash
cp .env.example .env
docker compose up --build -d
curl http://127.0.0.1:8787/health
```

Compose 会启动：

- `routetune`：管理台、统一 API、Router、Trace 和后台任务；
- `mock-provider`：三个本地 OpenAI-compatible 模型占位实现；
- `routetune-data`：SQLite 数据；
- `routetune-artifacts`：冷启动与训练产生的 Router 制品。

这里的双容器是零 Key 演示拓扑，不是 RouteTune 的强制部署架构：`routetune` 是唯一的
产品服务，`mock-provider` 只是一个独立的 OpenAI-compatible 测试 Provider。生产环境可
只运行 `routetune`，并在自己的 Compose/Kubernetes 配置中移除 `mock-provider` 及其
`depends_on`，连接真实云模型或内部 LiteLLM，同时从模型池停用三个 Mock 模型。

容器运行依赖由仓库中的 `requirements.lock` 固定，并使用哈希校验安装；该文件由
`uv.lock` 导出。更新 Python 依赖时必须同时更新并提交两份锁文件：

```bash
uv lock
uv export --frozen --no-dev --no-emit-project --output-file requirements.lock
```

首次启动时，容器自动创建 `tier_1`～`tier_5` 冷启动 Bundle。默认配置在回退层 `tier_5` 提供 `mock-strong`，所以不需要外部 Key 也能调用。发布版本同时包含真实模型与 Demo Mock 时，已配置凭据的真实 Provider 优先，Mock 只作为最后兜底；健康接口的 `provider_mode` 与 `routing_targets` 会显示当前实际首选。

## Key 配置

所有 Provider 统一使用环境变量。可以在启动命令前直接设置，也可以集中写进根目录 `.env`；Docker Compose 会自动读取该文件。

```dotenv
OPENROUTER_API_KEY=...
SILICONFLOW_API_KEY=...
DEEPSEEK_API_KEY=...
MINIMAX_API_KEY=...
ROUTETUNE_API_KEY=change-me
```

源码方式运行可使用 `routetune serve --env-file .env ...`。模型配置只保存 `api_key_env`，不会保存 Key 明文；Key 也不会进入 Release 或 Trace。

## 生产前必须修改

1. 设置足够长的随机 `ROUTETUNE_API_KEY`。
2. 在管理台加入真实 Provider 和模型，检测可用性并禁用 Mock 模型；自定义 OpenAI-compatible Provider 数量不受限制，以唯一 Provider ID 区分。
3. 创建候选 Release，确认模型池、安全策略和 Router Bundle 后人工发布。
4. 使用 Caddy、Nginx 或云负载均衡器终止 HTTPS。
5. 限制管理台网络访问；当前管理接口和业务接口共用一个 Bearer Token。
6. 建立 SQLite 和 Router artifacts 的定期备份。

默认端口只映射到宿主机 `127.0.0.1`。如需由另一台机器直接访问，可以修改 `compose.yaml` 的端口绑定，但推荐让反向代理与 RouteTune 位于同一主机或私有网络。

## 配置项

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `ROUTETUNE_PORT` | `8787` | 宿主机监听端口 |
| `ROUTETUNE_API_KEY` | 空 | 统一 Bearer Token；生产环境必须设置 |
| `ROUTETUNE_MODEL_CHECK_INTERVAL_SECONDS` | `1800` | 模型可用性检测周期；`0` 关闭 |
| `OPENROUTER_API_KEY` | 空 | OpenRouter Key |
| `SILICONFLOW_API_KEY` | 空 | 硅基流动 Key |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek Key |
| `MINIMAX_API_KEY` | 空 | MiniMax Key |
| `MINIMAX_BASE_URL` | 官方地址 | 可覆盖为兼容入口 |

## 健康检查与日志

```bash
docker compose ps
docker compose logs -f routetune
curl http://127.0.0.1:8787/health
```

模型池默认每 30 分钟顺序发送低 Token 请求检测启用模型；检测结果不写入业务 Trace。周期可以通过环境变量调整为 `900`（15 分钟）等值。

## 备份与恢复

一致的备份需要同时保存 SQLite 与 Router artifacts。单实例下建议短暂停止服务，然后备份 Compose 创建的两个 named volumes。实际 volume 名取决于项目目录，可先执行：

```bash
docker volume ls
```

确认名称后再使用临时容器导出。恢复时必须使用同一次备份中的数据库与 artifacts。

## 升级

```bash
git pull --ff-only
docker compose build --pull
docker compose up -d
```

升级前先备份。当前数据库迁移在服务启动时执行向后兼容的增量建表/加列，但开源 MVP 尚未承诺跨任意版本的自动迁移。

## 当前扩缩容边界

不要对当前 SQLite 版本直接增加多个 `routetune` 副本：后台批次、训练、定时模型检测和发布状态还没有分布式协调。

未来多副本版本需要共享事务数据库、外部任务队列、单例 Scheduler、Artifact 对象存储，以及无状态 API Runtime 与控制平面分离。这些方向见 [版本发布与扩缩容架构](RELEASE_ARCHITECTURE.md)，但不属于当前 MVP 的部署承诺。
