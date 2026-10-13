from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from routetune_core.contracts import Annotation, CandidateResult, EvaluationRecord


@dataclass(frozen=True)
class DemoModel:
    model_id: str
    cost_rank: int
    base_latency_ms: float
    success_by_scenario: dict[str, float]


MODELS = (
    DemoModel(
        model_id="economy",
        cost_rank=1,
        base_latency_ms=180,
        success_by_scenario={
            "simple": 0.95,
            "extract": 0.90,
            "summary": 0.78,
            "reasoning": 0.38,
            "code": 0.42,
            "legal": 0.28,
        },
    ),
    DemoModel(
        model_id="balanced",
        cost_rank=2,
        base_latency_ms=420,
        success_by_scenario={
            "simple": 0.985,
            "extract": 0.97,
            "summary": 0.94,
            "reasoning": 0.82,
            "code": 0.84,
            "legal": 0.72,
        },
    ),
    DemoModel(
        model_id="premium",
        cost_rank=3,
        base_latency_ms=850,
        success_by_scenario={
            "simple": 0.995,
            "extract": 0.995,
            "summary": 0.985,
            "reasoning": 0.97,
            "code": 0.97,
            "legal": 0.96,
        },
    ),
)


SCENARIOS = {
    "simple": (
        "请计算 {a} 加 {b}，只返回结果。",
        "把下面句子改得更简洁：我们目前正在进行相关工作的处理。编号 {n}",
        "将“项目 {n} 已完成”翻译成英文。",
    ),
    "extract": (
        "从订单中提取订单号和金额：订单 RT-{n:04d}，应付 {a}.{b:02d} 元。",
        "提取联系人姓名与电话：联系人张{surname}，电话 1380000{n:04d}。",
        "从文本中返回 JSON：设备 DEV-{n:04d} 在 {a} 点发生告警。",
    ),
    "summary": (
        "总结下面会议纪要的三个关键点：项目 {n} 延期，预算增加 {a}%，下周复盘风险。",
        "将这段客服记录概括为一句话：用户反馈版本 {n} 登录缓慢，多次重试后恢复。",
        "摘要这份周报：本周完成 {a} 个需求，修复 {b} 个缺陷，剩余风险是资源不足。",
    ),
    "reasoning": (
        "仓库有 {a} 箱零件，每箱 {b} 个，报废 {n_mod} 个，分给 3 个车间各多少？请推理。",
        "若服务 A 依赖 B，B 依赖 C，C 在 {a} 点故障，请分析最可能的影响链。案例 {n}",
        "给定规则：红色优先于蓝色，蓝色优先于绿色。对象 {n} 是绿色，应排在什么位置？说明理由。",
    ),
    "code": (
        "用 Python 编写函数，合并两个有序列表并去重。测试编号 {n}。",
        "分析这段 SQL 为什么产生重复行，并给出修复方案：SELECT * FROM a JOIN b ON a.id=b.id -- {n}",
        "修复并解释这个并发计数器的竞态条件，案例 {n}，线程数 {a}。",
    ),
    "legal": (
        "审查合同第 {a} 条违约责任，指出对乙方不利的风险。合同编号 HT-{n:04d}。",
        "比较保密条款和数据处理条款的责任边界，并给出修改建议。文件 {n}。",
        "判断以下自动续约条款是否存在明显风险，并说明依据：提前 {b} 天未通知即续约。",
    ),
}


def generate_demo_records(count: int = 600, seed: int = 42) -> list[EvaluationRecord]:
    if count < 30:
        raise ValueError("demo dataset requires at least 30 records")
    rng = random.Random(seed)
    scenario_names = tuple(SCENARIOS)
    records: list[EvaluationRecord] = []

    for index in range(count):
        scenario = scenario_names[index % len(scenario_names)]
        template = rng.choice(SCENARIOS[scenario])
        request_text = template.format(
            a=rng.randint(2, 99),
            b=rng.randint(2, 99),
            n=index,
            n_mod=index % 17,
            surname=rng.choice("王李赵陈周吴郑"),
        )
        request_id = f"demo-{index:05d}"
        candidates = tuple(
            _candidate_result(rng, request_id, request_text, scenario, model) for model in MODELS
        )
        baseline_model = "premium" if scenario in {"reasoning", "code", "legal"} else "balanced"
        records.append(
            EvaluationRecord(
                request_id=request_id,
                request_text=request_text,
                candidates=candidates,
                metadata={"scenario": scenario, "baseline_model_id": baseline_model},
            )
        )
    return records


def _candidate_result(
    rng: random.Random,
    request_id: str,
    request_text: str,
    scenario: str,
    model: DemoModel,
) -> CandidateResult:
    passed = rng.random() < model.success_by_scenario[scenario]
    length_factor = 1 + len(request_text) / 200
    latency = model.base_latency_ms * length_factor * rng.uniform(0.85, 1.15)
    output_hash = hashlib.sha256(f"{request_id}:{model.model_id}:{passed}".encode()).hexdigest()
    return CandidateResult(
        model_id=model.model_id,
        annotation=Annotation(
            passed=passed,
            failure_reason=None if passed else f"{scenario.upper()}_QUALITY_FAILURE",
            score=5.0 if passed else float(rng.randint(1, 3)),
        ),
        cost_rank=model.cost_rank,
        latency_ms=latency,
        output_hash=output_hash,
    )
