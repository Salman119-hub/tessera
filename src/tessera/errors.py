"""Exception hierarchy. Callers can catch ``TesseraError`` for anything raised here."""

from __future__ import annotations


class TesseraError(Exception):
    """Base class for all Tessera errors."""


class DefinitionError(TesseraError):
    """A feature definition is invalid or inconsistent with the repository."""


class SchemaError(TesseraError):
    """A record or frame does not match its declared schema."""


class ConflictError(TesseraError):
    """An optimistic-concurrency write lost the race; the caller should retry."""


class SkewGuardError(TesseraError):
    """A model was trained against feature definitions that differ from the live ones."""


class QualityCheckError(TesseraError):
    """A blocking data quality check failed."""

    def __init__(self, message: str, failures: list[str]) -> None:
        super().__init__(message)
        self.failures = failures
