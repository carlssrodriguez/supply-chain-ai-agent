from datetime import date

import pytest

from supply_chain_agent.approval import (
    ApprovalStore,
    DecisionAlreadyRecordedError,
    HumanDecision,
    InsufficientApprovalRoleError,
)
from supply_chain_agent.data_repository import SupplyChainRepository
from supply_chain_agent.replenishment import assess_replenishment


@pytest.fixture
def assessment():
    return assess_replenishment(
        "SKU-104",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )


def test_recommendation_survives_reopening_database(tmp_path, assessment) -> None:
    path = tmp_path / "approvals.db"
    created = ApprovalStore(path).create(assessment)

    loaded = ApprovalStore(path).get(created.recommendation_id)

    assert loaded.assessment.product_id == "SKU-104"
    assert loaded.assessment.recommended_order_quantity == 500
    assert loaded.decision is None


def test_authorized_human_can_approve(tmp_path, assessment) -> None:
    store = ApprovalStore(tmp_path / "approvals.db")
    created = store.create(assessment)

    decided = store.record_decision(
        created.recommendation_id,
        HumanDecision(
            decision="approved",
            approver_name="Alex Morgan",
            approver_role="Procurement Manager",
            rationale="Inventory and supplier data were manually verified.",
        ),
    )

    assert decided.decision is not None
    assert decided.decision.decision == "approved"
    assert decided.decided_at is not None


def test_role_below_policy_threshold_cannot_approve(tmp_path, assessment) -> None:
    assessment.required_approver = "Director of Supply Chain"
    store = ApprovalStore(tmp_path / "approvals.db")
    created = store.create(assessment)

    with pytest.raises(InsufficientApprovalRoleError, match="Director"):
        store.record_decision(
            created.recommendation_id,
            HumanDecision(
                decision="approved",
                approver_name="Alex Morgan",
                approver_role="Procurement Manager",
                rationale="Approved.",
            ),
        )


def test_rejection_does_not_require_minimum_approval_role(tmp_path, assessment) -> None:
    assessment.required_approver = "Chief Operating Officer"
    store = ApprovalStore(tmp_path / "approvals.db")
    created = store.create(assessment)

    decided = store.record_decision(
        created.recommendation_id,
        HumanDecision(
            decision="rejected",
            approver_name="Alex Morgan",
            approver_role="Procurement Manager",
            rationale="The inventory snapshot must be refreshed first.",
        ),
    )

    assert decided.decision is not None
    assert decided.decision.decision == "rejected"


def test_decision_cannot_be_changed_after_recording(tmp_path, assessment) -> None:
    store = ApprovalStore(tmp_path / "approvals.db")
    created = store.create(assessment)
    first_decision = HumanDecision(
        decision="rejected",
        approver_name="Alex Morgan",
        approver_role="Procurement Manager",
        rationale="Refresh the data.",
    )
    store.record_decision(created.recommendation_id, first_decision)

    with pytest.raises(DecisionAlreadyRecordedError):
        store.record_decision(created.recommendation_id, first_decision)
