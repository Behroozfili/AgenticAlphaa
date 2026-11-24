from agents.executor_helpers.monthly_forecast_helper import assemble_monthly_forecast
from agents.state import MonthlyForecastSlot
from api.routes.analyze import MonthlyForecastResponse, _compact_monthly_forecast
from tests.unit_tests.test_stage8_manager_integration import complete_state


def test_state_result_and_api_monthly_forecast_schemas_are_identical():
    result = assemble_monthly_forecast(complete_state()).to_dict()
    typed_fields = set(MonthlyForecastSlot.__annotations__)
    api_fields = set(MonthlyForecastResponse.model_fields)

    assert set(result) == typed_fields == api_fields
    assert {"uncertainty_low", "uncertainty_high", "model_version"}.isdisjoint(typed_fields)
    assert {
        "confidence_interval", "versions", "status", "target_date", "warnings",
        "cutoff_policy", "quality_gate_passed", "quality_gate_failures",
        "narrative", "implementation",
    } <= typed_fields


def test_compact_api_serializer_keeps_exact_contract_and_validates():
    result = assemble_monthly_forecast(complete_state()).to_dict()
    compact = _compact_monthly_forecast({"monthly_forecast": result})

    assert compact is not None
    assert set(compact) == set(MonthlyForecastResponse.model_fields)
    assert MonthlyForecastResponse.model_validate(compact).model_dump() == compact

