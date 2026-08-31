from ai_data_engineering.environment import build_environment_report


def test_environment_report_has_no_local_paths() -> None:
    report = build_environment_report()

    assert report["schema_version"] == "0.1"
    assert "python" in report
    assert "platform" in report
    assert "capabilities" in report
    assert "cwd" not in report
    assert "home" not in report
