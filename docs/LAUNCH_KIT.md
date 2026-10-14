# RouteTune Launch Kit

## 一句话

RouteTune 是可私有化部署的 LLM 路由闭环：用企业自己的请求、模型实测和人工反馈训练轻量分类 Router，把每条请求交给“刚好够用”的模型。

英文版本：

> RouteTune is a self-hosted feedback loop that learns a lightweight model router from your real prompts, candidate outputs, and human feedback—then serves it behind one OpenAI-compatible API.

## 30 秒介绍

很多团队要么把所有请求都交给昂贵的强模型，要么维护一堆很快失效的人工规则。RouteTune 提供另一条路：导入真实请求，让多个候选模型作答，快速标注哪些结果可接受，然后训练一个输出固定能力层级及概率的非生成式 Router。通过人工发布后，业务仍然只调用一个 OpenAI-compatible API；线上 Trace 和失败反馈会进入下一轮训练。

RouteTune 可以部署在自己的服务器或云账号中。控制面、Router、Trace 和标注数据由企业持有；既可以对接内网 LiteLLM/私有模型形成完整私有链路，也可以在出站脱敏后调用指定的外部 Provider。RouteTune 的重点不是“评测更多模型”，而是建立一条安全可控、持续改进、可回滚、可追溯的业务路由闭环。

## GitHub Description

```text
Self-hosted, privacy-conscious feedback loop for training and serving probabilistic LLM routers behind one OpenAI-compatible API.
```

建议 Topics：

```text
llm-routing, model-router, llm-gateway, openai-compatible, llmops, machine-learning, self-hosted
```

## 中文首发帖

```text
我们开源了 RouteTune：一个可私有化部署、用真实业务反馈训练 LLM Router 的自托管工具。

它不让另一个 LLM 决定“该调用哪个 LLM”，也不依赖一套通用 benchmark。你导入自己的请求，运行多个候选模型，标注哪些输出可接受，RouteTune 再训练一个轻量概率分类器，把新请求路由到刚好够用的模型层级。

整个流程是一条闭环：
路由数据 → 候选对比 → 人工标注 → Router 训练 → Router 版本 → 统一 API → Trace 回流。

首个开源版本包含模型池、双向安全脱敏、路由数据、人工标注、Router 版本、调用大盘和 OpenAI-compatible API。控制面、Router、Trace 和标注数据可以留在自己的基础设施里；接内部 LiteLLM 或私有模型时，完整调用链都可以运行在内网。

Docker Compose 可以零 Key 启动，本地 Mock 模型能直接跑完整流程。Provider Key 只通过环境变量注入，不进入模型配置、Release 或 Trace。

GitHub: https://github.com/HuggingAI/RouteTune

目前是单机 MVP。我们最想听到的不是“再接几个模型”，而是：在你的业务里，什么样的请求路由才算真的成功？
```

## 英文首发帖

```text
We are open-sourcing RouteTune: a self-hosted, privacy-conscious feedback loop for LLM routing.

Instead of asking another LLM to choose a model—or trusting a generic benchmark—RouteTune learns from your real prompts, candidate outputs, and pass/fail feedback. It trains a lightweight probabilistic classifier that routes each request to the lowest capability tier likely to meet your quality bar.

The loop is simple:
import prompts → compare models → review outputs → train → publish → serve through one OpenAI-compatible API → learn from production traces.

The first release includes a model pool, bidirectional redaction, batch runs, a focused review workflow, versioned releases, operational metrics, and a unified API. The control plane, router artifacts, traces, and annotations stay in infrastructure you control. With an internal LiteLLM or private model endpoint, the complete request path can remain inside your network.

Docker Compose starts a complete zero-key demo with local mock models. Provider credentials are injected through environment variables and are never stored in model configuration, releases, or traces.

GitHub: https://github.com/HuggingAI/RouteTune

This is an early single-node MVP. We would especially love feedback from teams with real routing decisions—not just benchmark leaderboards.
```

## 可用于宣传的安全事实

- Self-hosted：控制面、Router、Trace 和标注数据由部署方持有；
- Private-model ready：内部 LiteLLM 和 OpenAI-compatible 私有端点与外部 Provider 平级；
- Bidirectional redaction：Provider 调用前和业务返回前都执行敏感信息扫描；
- Minimal trace：Trace 只保存脱敏内容，也可只保存 metadata；
- Secret separation：Key 只从环境变量或 `.env` 读取，不进入版本制品和日志；
- Controlled rollout：配置修改必须生成不可变 Release 并人工发布，支持回滚和追溯。

建议用“安全可控”或“privacy-conscious”，不要使用“绝对安全”“零泄露”或“合规开箱即用”。是否有请求离开私有网络，取决于部署方选择内部模型还是外部 Provider。

## 发布素材清单

首发前建议准备三张真实产品截图：

1. API 运行大盘：模型分布、成功率、延迟、Input/Output Token；
2. 统一审阅页：同一请求的 GT、多个模型输出和 pass/fail；
3. 发布页：候选 Router、不可变 Release 和当前 Production。

截图使用本地 Mock 或公开样例数据，隐藏 API Key、内部域名、客户输入、Trace ID 和本机路径。三张图应讲同一条产品闭环，不要为每个功能单独做海报。

## Demo 顺序（3～5 分钟）

1. 在 Playground 发一条请求，展示统一 API 和 Trace。
2. 导入 10 条样例数据，选择模型对比模式并运行。
3. 在统一审阅页快速完成几条 pass/fail。
4. 选择训练器并生成候选 Router。
5. 创建 Release、查看变更摘要并发布。
6. 再发一条请求，回到大盘说明模型选择和版本可追溯。

## 不要这样宣传

- 不声称已经实现多副本、高可用或大规模分布式训练；
- 不承诺固定节省比例，节省取决于业务请求和候选模型；
- 不把它描述成通用评测平台或完整 LLM Gateway；
- 不使用用户提供的真实敏感数据做公开截图；
- 不把 Jev 描述成新算法；RouteTune 的 Router 本质是带概率的分类模型。
- 不把内置脱敏描述成完整 DLP、合规认证或数据绝不出网。
