from fastapi.testclient import TestClient

from routetune_core.mock_litellm import create_mock_litellm_app


def test_mock_litellm_exposes_models_and_openai_completion() -> None:
    client = TestClient(create_mock_litellm_app())

    models = client.get("/v1/models")
    completion = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-balanced",
            "messages": [{"role": "user", "content": "评测这条链路"}],
            "temperature": 0.2,
        },
    )

    assert models.status_code == 200
    assert {item["id"] for item in models.json()["data"]} == {
        "mock-fast",
        "mock-balanced",
        "mock-strong",
    }
    assert completion.status_code == 200
    assert completion.json()["model"] == "mock-balanced"
    assert "RouteTune" in completion.json()["choices"][0]["message"]["content"]
