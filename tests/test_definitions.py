from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera import Aggregate, AggregateFeatureView, FeatureService, Field
from tessera.definitions import Check, default_hop
from tessera.errors import DefinitionError
from tessera.repository import Repository, load_registry
from tessera.timeutil import align, format_duration, parse_duration, to_epoch_seconds

from .conftest import EXAMPLE, clicks, service, user, user_clicks, user_profile


class TestTime:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("30s", 30), ("5m", 300), ("1h", 3600), ("7d", 604_800), ("2w", 1_209_600)],
    )
    def test_parse_and_format(self, text: str, seconds: int) -> None:
        assert parse_duration(text) == seconds
        assert parse_duration(format_duration(seconds)) == seconds

    @pytest.mark.parametrize("bad", ["", "5", "5x", "-1m", "0m", 0, True])
    def test_rejects_invalid(self, bad: object) -> None:
        with pytest.raises(ValueError):
            parse_duration(bad)  # type: ignore[arg-type]

    def test_align_and_epoch(self) -> None:
        assert align(3_725, 3600) == 3600
        assert to_epoch_seconds("1970-01-01T01:00:00Z") == 3600
        assert to_epoch_seconds("1970-01-01T01:00:00") == 3600  # naive means UTC
        with pytest.raises(TypeError):
            to_epoch_seconds(True)


class TestDefinitions:
    def test_default_hop_policy(self) -> None:
        assert default_hop(600, 60) == 60
        assert default_hop(86_400, 60) == 3600
        assert default_hop(30 * 86_400, 300) == 86_400

    def test_feature_names_and_qualified(self) -> None:
        assert user_clicks.feature_names[0] == "events_count_10m"
        assert user_clicks.qualified("value_sum_1h") == "user_clicks__value_sum_1h"
        assert service.output_columns[-2:] == [
            "click_context__price_vs_mean",
            "click_context__has_history",
        ]

    def test_dtypes(self) -> None:
        assert user_clicks.feature_dtype("events_count_10m") == "int"
        assert user_clicks.feature_dtype("item_last_1d") == "string"
        assert user_clicks.feature_dtype("value_mean_1d") == "float"

    @pytest.mark.parametrize("name", ["Bad", "1abc", "a__b", ""])
    def test_rejects_bad_names(self, name: str) -> None:
        with pytest.raises(DefinitionError):
            Field(name)

    def test_window_must_divide_by_hop(self) -> None:
        with pytest.raises(DefinitionError, match="not a multiple"):
            AggregateFeatureView(
                "v",
                user,
                clicks,
                aggregates=(Aggregate("value", "sum", "90s", hop="1m"),),
                granularity="1m",
            )

    def test_unknown_column_and_function(self) -> None:
        with pytest.raises(DefinitionError, match="unknown column"):
            AggregateFeatureView("v", user, clicks, aggregates=(Aggregate("nope", "sum", "1h"),))
        with pytest.raises(DefinitionError, match="unknown aggregation"):
            Aggregate("value", "median", "1h")  # type: ignore[arg-type]
        with pytest.raises(DefinitionError, match="needs a column"):
            Aggregate(None, "sum", "1h")

    def test_service_validation(self) -> None:
        with pytest.raises(DefinitionError, match="twice"):
            FeatureService("s", features=(user_clicks, user_clicks.select("value_sum_1h")))
        with pytest.raises(DefinitionError, match="no features"):
            user_profile.select("missing")
        from .conftest import click_context

        with pytest.raises(DefinitionError, match="depends on"):
            FeatureService("s", features=(click_context,))

    def test_check_on_unknown_column(self) -> None:
        from tessera import BatchSource

        with pytest.raises(DefinitionError, match="unknown column"):
            BatchSource("s", "p.csv", "ts", (Field("a"),), checks=(Check.not_null("b"),))

    def test_fingerprint_changes_with_definition_but_not_identity(self) -> None:
        same = AggregateFeatureView(
            "user_clicks", user, clicks, granularity="1m", aggregates=user_clicks.aggregates
        )
        assert same.fingerprint == user_clicks.fingerprint
        changed = AggregateFeatureView(
            "user_clicks",
            user,
            clicks,
            granularity="1m",
            aggregates=(*user_clicks.aggregates, Aggregate("value", "sum", "1d")),
        )
        assert changed.fingerprint != user_clicks.fingerprint
        assert changed.namespace != user_clicks.namespace

    def test_transform_body_is_part_of_fingerprint(self) -> None:
        def t1(row: dict) -> dict:  # type: ignore[type-arg]
            return {**row, "x": 1.0}

        def t2(row: dict) -> dict:  # type: ignore[type-arg]
            return {**row, "x": 2.0}

        a = AggregateFeatureView(
            "v", user, clicks, transform=t1, aggregates=(Aggregate("x", "sum", "1h"),)
        )
        b = AggregateFeatureView(
            "v", user, clicks, transform=t2, aggregates=(Aggregate("x", "sum", "1h"),)
        )
        assert a.fingerprint != b.fingerprint


class TestRepository:
    def test_collects_dependencies(self, repo: Repository) -> None:
        assert set(repo.views) == {"user_clicks", "user_profile", "click_context"}
        assert set(repo.sources) == {"clicks", "profiles"}
        assert repo.aggregate_views_for_source("clicks") == [user_clicks]
        with pytest.raises(DefinitionError):
            repo.view("nope")
        with pytest.raises(DefinitionError):
            repo.service("nope")

    def test_conflicting_names(self) -> None:
        other = AggregateFeatureView(
            "user_clicks", user, clicks, aggregates=(Aggregate("value", "sum", "1h"),)
        )
        with pytest.raises(DefinitionError, match="two different"):
            Repository([user_clicks, other])

    def test_plan_and_apply(self, repo: Repository, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        first = repo.apply(registry)
        assert {c.name for c in first.changes if c.action == "add"} >= {"user_clicks", "recs_v1"}
        assert sorted(first.backfills) == ["user_clicks", "user_profile"]
        assert repo.plan(load_registry(registry)).empty
        assert "No changes" in repo.plan(load_registry(registry)).render()

        changed_view = AggregateFeatureView(
            "user_clicks",
            user,
            clicks,
            granularity="1m",
            aggregates=(*user_clicks.aggregates[:-1], Aggregate("value", "min", "2d")),
        )
        new_service = FeatureService("recs_v1", features=(changed_view,))
        plan = Repository([new_service]).plan(load_registry(registry))
        by_name = {c.name: c for c in plan.changes}
        assert by_name["user_clicks"].action == "modify"
        assert by_name["user_clicks"].requires_backfill
        assert "feature +value_min_2d" in by_name["user_clicks"].details
        assert "feature -value_min_3d" in by_name["user_clicks"].details
        assert by_name["recs_v1"].action == "modify"
        assert by_name["user_profile"].action == "remove"
        rendered = plan.render()
        assert "~ view user_clicks  [backfill required]" in rendered

    def test_from_path_loads_example(self, example_repo: Repository) -> None:
        assert "fraud_v1" in example_repo.services
        assert len(example_repo.service("fraud_v1").output_columns) == 23

    def test_from_path_errors(self, tmp_path: Path) -> None:
        with pytest.raises(DefinitionError, match="not found"):
            Repository.from_path(tmp_path / "missing.py")
        broken = tmp_path / "broken.py"
        broken.write_text("raise RuntimeError('boom')\n")
        with pytest.raises(DefinitionError, match="boom"):
            Repository.from_path(broken)

    def test_registry_version_check(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"version": 42}))
        with pytest.raises(DefinitionError, match="unsupported"):
            load_registry(path)
        assert load_registry(tmp_path / "absent.json") is None


def test_example_definitions_file_exists() -> None:
    assert (EXAMPLE / "features.py").exists()
