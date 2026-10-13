from routetune_core.security import DictionaryDetector, SecurityPipeline


def test_security_redacts_regex_and_dictionary_entities() -> None:
    pipeline = SecurityPipeline(
        detectors=[
            *SecurityPipeline().detectors,
            DictionaryDetector({"CUSTOMER": ["星河公司"]}),
        ]
    )

    result = pipeline.redact("联系星河公司：13800138000 或 boss@example.com")

    assert "星河公司" not in result.text
    assert "13800138000" not in result.text
    assert "boss@example.com" not in result.text
    assert result.counts == {"CUSTOMER": 1, "PHONE": 1, "EMAIL": 1}
    assert all("text" not in item.to_safe_dict() for item in result.findings)
