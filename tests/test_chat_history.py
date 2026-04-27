from backend.generator import select_relevant_history


def test_keep_history_for_follow_up_query() -> None:
    history = [
        {"user": "Quy uoc danh so thiet bi la gi?", "assistant": "Tra loi A"},
        {"user": "Cap 220kV dung so nao?", "assistant": "So 2"},
    ]
    kept = select_relevant_history("vay 500kV thi sao?", history)
    assert len(kept) == 2


def test_drop_history_when_topic_changes() -> None:
    history = [
        {"user": "Quy uoc danh so thiet bi la gi?", "assistant": "Tra loi A"},
        {"user": "Cap 220kV dung so nao?", "assistant": "So 2"},
    ]
    kept = select_relevant_history("Quy trinh nghi phep nhu the nao?", history)
    assert kept == []
