"""Persistent human-approval workflow for draft replenishment recommendations."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from supply_chain_agent.replenishment import ReplenishmentAssessment


ApprovalRole = Literal[
    "Procurement Manager",
    "Director of Supply Chain",
    "Chief Operating Officer",
]
Decision = Literal["approved", "rejected"]

ROLE_LEVELS: dict[str, int] = {
    "Procurement Manager": 1,
    "Director of Supply Chain": 2,
    "Chief Operating Officer": 3,
}


class RecommendationNotFoundError(LookupError):
    """Raised when a recommendation identifier does not exist."""


class DecisionAlreadyRecordedError(RuntimeError):
    """Raised when someone tries to decide the same recommendation twice."""


class InsufficientApprovalRoleError(PermissionError):
    """Raised when the approver is below the policy's minimum role."""


class HumanDecision(BaseModel):
    decision: Decision
    approver_name: str = Field(min_length=1)
    approver_role: ApprovalRole
    rationale: str = Field(min_length=1)


class RecommendationRecord(BaseModel):
    recommendation_id: str
    created_at: datetime
    assessment: ReplenishmentAssessment
    decision: HumanDecision | None = None
    decided_at: datetime | None = None


class ApprovalStore:
    """Store immutable recommendation decisions in a local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(self, assessment: ReplenishmentAssessment) -> RecommendationRecord:
        record = RecommendationRecord(
            recommendation_id=str(uuid4()),
            created_at=datetime.now(UTC),
            assessment=assessment,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recommendations (
                    recommendation_id, created_at, assessment_json
                ) VALUES (?, ?, ?)
                """,
                (
                    record.recommendation_id,
                    record.created_at.isoformat(),
                    assessment.model_dump_json(),
                ),
            )
        return record

    def get(self, recommendation_id: str) -> RecommendationRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT recommendation_id, created_at, assessment_json,
                       decision_json, decided_at
                FROM recommendations
                WHERE recommendation_id = ?
                """,
                (recommendation_id,),
            ).fetchone()

        if row is None:
            raise RecommendationNotFoundError(recommendation_id)
        return self._row_to_record(row)

    def record_decision(
        self,
        recommendation_id: str,
        decision: HumanDecision,
    ) -> RecommendationRecord:
        current = self.get(recommendation_id)
        if current.decision is not None:
            raise DecisionAlreadyRecordedError(recommendation_id)
        if decision.decision == "approved":
            self._validate_approval_role(current.assessment, decision.approver_role)

        decided_at = datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE recommendations
                SET decision_json = ?, decided_at = ?
                WHERE recommendation_id = ? AND decision_json IS NULL
                """,
                (
                    decision.model_dump_json(),
                    decided_at.isoformat(),
                    recommendation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DecisionAlreadyRecordedError(recommendation_id)

        return self.get(recommendation_id)

    def _validate_approval_role(
        self,
        assessment: ReplenishmentAssessment,
        approver_role: ApprovalRole,
    ) -> None:
        required_role = assessment.required_approver
        if required_role is None:
            raise InsufficientApprovalRoleError(
                "A recommendation with no order value cannot be approved."
            )
        if ROLE_LEVELS[approver_role] < ROLE_LEVELS[required_role]:
            raise InsufficientApprovalRoleError(
                f"{required_role} or a higher role is required."
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    assessment_json TEXT NOT NULL,
                    decision_json TEXT,
                    decided_at TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @staticmethod
    def _row_to_record(row: tuple[str, str, str, str | None, str | None]) -> RecommendationRecord:
        recommendation_id, created_at, assessment_json, decision_json, decided_at = row
        return RecommendationRecord(
            recommendation_id=recommendation_id,
            created_at=datetime.fromisoformat(created_at),
            assessment=ReplenishmentAssessment.model_validate_json(assessment_json),
            decision=(
                HumanDecision.model_validate(json.loads(decision_json))
                if decision_json
                else None
            ),
            decided_at=datetime.fromisoformat(decided_at) if decided_at else None,
        )
