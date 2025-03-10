"""Data quality checks on source frames.

Checks are declared on the source definition and run before any batch
materialisation, backfill or training-set build. ``mostly`` allows a bounded
failure fraction (the Great Expectations convention), and ``severity="warn"``
records a failure without blocking.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from tessera.definitions import BatchSource, Check
from tessera.errors import QualityCheckError


@dataclass(frozen=True)
class CheckResult:
    check: str
    column: str
    severity: str
    success: bool
    evaluated: int
    failed: int
    observed_fraction: float
    required_fraction: float

    def describe(self) -> str:
        return (
            f"{self.check}({self.column}): {self.failed}/{self.evaluated} failing rows, "
            f"pass rate {self.observed_fraction:.4f} < required {self.required_fraction:.4f}"
        )


@dataclass(frozen=True)
class QualityReport:
    source: str
    row_count: int
    results: tuple[CheckResult, ...]

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.success and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.success and r.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            failures = [r.describe() for r in self.errors]
            raise QualityCheckError(
                f"source {self.source!r} failed {len(failures)} blocking check(s): "
                + "; ".join(failures),
                failures,
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "row_count": self.row_count,
            "passed": self.passed,
            "results": [asdict(r) for r in self.results],
        }


def _failing_mask(df: pd.DataFrame, check: Check) -> tuple[pd.Series, pd.Series]:
    """Return (evaluated mask, failing mask). Nulls are only evaluated by not_null."""
    col = df[check.column]
    present = col.notna()
    if check.kind == "not_null":
        return pd.Series(True, index=df.index), ~present
    if check.kind == "between":
        low, high = check.args
        values = pd.to_numeric(col, errors="coerce")
        bad = pd.Series(False, index=df.index)
        if low is not None:
            bad |= values < low
        if high is not None:
            bad |= values > high
        bad |= present & values.isna()  # non-numeric
        return present, bad & present
    if check.kind == "in_set":
        allowed = set(check.args)
        return present, present & ~col.isin(allowed)
    if check.kind == "unique":
        return present, present & col.duplicated(keep=False)
    if check.kind == "regex":
        pattern = re.compile(check.args[0])
        bad = col.map(lambda v: v is not None and not pattern.fullmatch(str(v)))
        return present, present & bad.astype(bool)
    raise ValueError(f"unknown check kind {check.kind!r}")


def run_checks(df: pd.DataFrame, source: BatchSource) -> QualityReport:
    results: list[CheckResult] = []
    for check in source.checks:
        evaluated_mask, failing_mask = _failing_mask(df, check)
        evaluated = int(evaluated_mask.sum())
        failed = int((failing_mask & evaluated_mask).sum())
        fraction = 1.0 if evaluated == 0 else (evaluated - failed) / evaluated
        results.append(
            CheckResult(
                check=check.kind,
                column=check.column,
                severity=check.severity,
                success=fraction >= check.mostly,
                evaluated=evaluated,
                failed=failed,
                observed_fraction=round(fraction, 6),
                required_fraction=check.mostly,
            )
        )
    return QualityReport(source=source.name, row_count=len(df), results=tuple(results))
