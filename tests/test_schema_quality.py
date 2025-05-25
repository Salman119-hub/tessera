from __future__ import annotations

import pandas as pd
import pytest

from tessera import BatchSource, Field
from tessera.definitions import Check
from tessera.errors import QualityCheckError, SchemaError
from tessera.quality import run_checks
from tessera.schema import TS_COLUMN, coerce_frame, coerce_record, coerce_value

from .conftest import T0, clicks

source = BatchSource(
    name="orders",
    path="unused.csv",
    timestamp_field="ts",
    schema=(
        Field("order_id", "string"),
        Field("qty", "int"),
        Field("price", "float"),
        Field("express", "bool"),
        Field("country", "string"),
    ),
    checks=(
        Check.not_null("order_id"),
        Check.unique("order_id"),
        Check.between("price", 0, 100),
        Check.in_set("country", ["US", "GB"], mostly=0.6),
        Check.regex("country", r"[A-Z]{2}", severity="warn"),
        Check.not_null("qty", mostly=0.5, severity="warn"),
    ),
)


class TestCoercion:
    @pytest.mark.parametrize(
        ("value", "dtype", "expected"),
        [
            ("3", "int", 3),
            (3.0, "int", 3),
            ("2.5", "float", 2.5),
            ("yes", "bool", True),
            ("F", "bool", False),
            (0, "bool", False),
            (12, "string", "12"),
            (None, "float", None),
            (float("nan"), "int", None),
        ],
    )
    def test_coerce_value(self, value: object, dtype: str, expected: object) -> None:
        assert coerce_value(value, dtype) == expected

    @pytest.mark.parametrize(
        ("value", "dtype"),
        [(2.5, "int"), ("maybe", "bool"), ("x", "float"), (float("inf"), "float")],
    )
    def test_coerce_value_errors(self, value: object, dtype: str) -> None:
        with pytest.raises(ValueError):
            coerce_value(value, dtype)

    def test_coerce_record(self) -> None:
        out = coerce_record(
            {
                "event_id": 7,
                "user_id": "u",
                "value": "1.5",
                "item": "a",
                "ts": "2026-01-01T00:00:00Z",
                "extra": "dropped",
            },
            clicks,
        )
        assert out["event_id"] == "7"
        assert out["value"] == 1.5
        assert out[TS_COLUMN] == 1_767_225_600
        assert "extra" not in out

    def test_coerce_record_collects_all_errors(self) -> None:
        with pytest.raises(SchemaError) as info:
            coerce_record({"event_id": "1", "user_id": "u", "value": "abc"}, clicks)
        message = str(info.value)
        assert "value" in message
        assert "missing timestamp" in message

    def test_coerce_frame(self) -> None:
        frame = pd.DataFrame(
            {
                "order_id": [1, 2],
                "qty": ["1", None],
                "price": ["2.5", "3"],
                "express": ["true", "no"],
                "country": ["US", "GB"],
                "ts": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
            }
        )
        out = coerce_frame(frame, source)
        assert out[TS_COLUMN].tolist() == [1_767_225_600, 1_767_225_601]
        assert out["order_id"].tolist() == ["1", "2"]
        assert out["express"].tolist() == [True, False]
        assert str(out["qty"].dtype) == "Int64"

    def test_coerce_frame_epoch_and_errors(self) -> None:
        frame = pd.DataFrame(
            {
                "order_id": ["a"],
                "qty": [1],
                "price": [1.0],
                "express": [True],
                "country": ["US"],
                "ts": [T0],
            }
        )
        assert coerce_frame(frame, source)[TS_COLUMN].tolist() == [T0]
        with pytest.raises(SchemaError, match="missing columns"):
            coerce_frame(frame.drop(columns=["price"]), source)
        with pytest.raises(SchemaError, match="price"):
            coerce_frame(frame.assign(price=["oops"]), source)
        with pytest.raises(SchemaError, match="without a timestamp"):
            coerce_frame(frame.assign(ts=[None]), source)


class TestQuality:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "order_id": ["a", "b", "b", None],
                "qty": [1, None, None, 2],
                "price": [10.0, 150.0, 20.0, None],
                "country": ["US", "GB", "fr", "DE"],
            }
        )

    def test_results(self) -> None:
        report = run_checks(self.frame(), source)
        by_kind = {(r.check, r.column): r for r in report.results}
        assert by_kind[("not_null", "order_id")].failed == 1
        assert not by_kind[("unique", "order_id")].success
        between = by_kind[("between", "price")]
        assert (between.evaluated, between.failed) == (3, 1)  # nulls are not range-checked
        in_set = by_kind[("in_set", "country")]
        assert in_set.observed_fraction == 0.5 and not in_set.success
        assert not by_kind[("regex", "country")].success
        assert by_kind[("not_null", "qty")].success  # 50% present meets mostly=0.5
        assert len(report.warnings) == 1
        assert not report.passed
        assert report.to_dict()["row_count"] == 4

    def test_raise_for_errors(self) -> None:
        report = run_checks(self.frame(), source)
        with pytest.raises(QualityCheckError) as info:
            report.raise_for_errors()
        assert len(info.value.failures) == len(report.errors)

    def test_clean_frame_passes(self) -> None:
        clean = pd.DataFrame(
            {"order_id": ["a", "b"], "qty": [1, 2], "price": [1.0, 2.0], "country": ["US", "GB"]}
        )
        report = run_checks(clean, source)
        assert report.passed and not report.warnings
        report.raise_for_errors()
