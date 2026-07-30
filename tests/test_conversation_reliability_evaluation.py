from pathlib import Path

from app.conversation.models import MessageRole
from app.conversation.reliability_evaluation import evaluate_summary_drift
from app.conversation.store import SqliteConversationStore


def test_summary_drift_evaluation_preserves_release_and_rejects_invention(
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        message = store.add_message(
            conversation.conversation_id,
            MessageRole.USER,
            "For R24, compare T-1 with B-04.",
        )

        faithful = evaluate_summary_drift(
            previous_summary=None,
            messages=[message],
            candidate_summary="The user asks about R24 T-1 and B-04.",
        )
        drifted = evaluate_summary_drift(
            previous_summary=None,
            messages=[message],
            candidate_summary="The user asks about R25 T-1 and B-09.",
        )
        lost_scope = evaluate_summary_drift(
            previous_summary=None,
            messages=[message],
            candidate_summary="The user asks about teller and branch reports.",
        )

    assert faithful.passed is True
    assert drifted.passed is False
    assert drifted.invented_anchors == ("B-09", "R25")
    assert drifted.missing_release_anchors == ("R24",)
    assert lost_scope.passed is False
    assert lost_scope.missing_release_anchors == ("R24",)
