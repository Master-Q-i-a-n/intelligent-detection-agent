from collections import deque

import pytest

from safety_operations.monitor import (
    ABSENT,
    HELMET,
    NO_HELMET,
    UNKNOWN,
    Detection,
    EventState,
    TrackState,
    add_helmet_observation,
    advance_event,
    associate_ppe,
    positive_evidence_duration,
    update_zone_membership,
)


def make_track() -> TrackState:
    return TrackState(
        track_id=1,
        first_seen_at=0.0,
        last_seen_at=0.0,
        bbox=(0.0, 0.0, 100.0, 200.0),
        person_confidence=0.9,
        foot_point=(50.0, 200.0),
        head_roi=(0.0, 0.0, 100.0, 70.0),
    )


def test_zone_entry_and_exit_require_confirmation() -> None:
    state = make_track()

    update_zone_membership(state, "work_area", 0.0, 0.5, 0.8)
    assert state.current_zone is None
    update_zone_membership(state, "work_area", 0.6, 0.5, 0.8)
    assert state.current_zone == "work_area"
    assert state.zone_enter_at == 0.6

    update_zone_membership(state, None, 1.0, 0.5, 0.8)
    assert state.current_zone == "work_area"
    update_zone_membership(state, None, 1.9, 0.5, 0.8)
    assert state.current_zone is None
    assert state.zone_enter_at is None


def test_ppe_association_uses_upper_region_and_is_one_to_one() -> None:
    persons = [
        Detection(6, (0.0, 0.0, 100.0, 200.0), 0.9, 1),
        Detection(6, (60.0, 0.0, 160.0, 200.0), 0.9, 2),
    ]
    helmets = [
        Detection(0, (70.0, 10.0, 90.0, 30.0), 0.9),
        # 位于人物框下部，不应被视为佩戴。
        Detection(0, (20.0, 120.0, 40.0, 145.0), 0.9),
    ]

    matches = associate_ppe(persons, helmets, [], 0.35)
    matched_count = sum(value["helmet"] is not None for value in matches.values())
    assert matched_count == 1
    assert matches[1]["helmet"] is not None
    assert matches[2]["helmet"] is None


def test_conflicting_helmet_and_no_helmet_can_be_marked_unknown() -> None:
    person = Detection(6, (0.0, 0.0, 100.0, 200.0), 0.9, 1)
    helmet = Detection(0, (35.0, 10.0, 65.0, 35.0), 0.8)
    no_helmet = Detection(7, (38.0, 12.0, 62.0, 38.0), 0.7)

    matches = associate_ppe([person], [helmet], [no_helmet], 0.35)
    observation = (
        UNKNOWN
        if matches[1]["helmet"] is not None
        and matches[1]["no_helmet"] is not None
        else HELMET
    )
    assert observation == UNKNOWN


def test_gloves_use_body_box_and_goggles_use_head_roi() -> None:
    person = Detection(6, (0.0, 0.0, 100.0, 200.0), 0.9, 1)
    glove = Detection(1, (70.0, 120.0, 90.0, 150.0), 0.8)
    goggles = Detection(4, (35.0, 18.0, 65.0, 35.0), 0.8)
    matches = associate_ppe(
        [person], [], [], 0.35, [glove], [], [goggles], []
    )

    assert matches[1]["gloves"] == [glove]
    assert matches[1]["goggles"] == goggles


def test_positive_evidence_allows_intermittent_hits_but_requires_two_frames() -> None:
    history = deque([(1.0, 0.8)])
    frames, duration = positive_evidence_duration(history, 1.0, 3.0, 0.5)
    assert frames == 1
    assert duration == 0.5

    history.append((1.4, 0.7))
    frames, duration = positive_evidence_duration(history, 1.4, 3.0, 0.5)
    assert frames == 2
    assert duration == pytest.approx(0.9)

    frames, _ = positive_evidence_duration(history, 4.5, 3.0, 0.5)
    assert frames == 0


def test_helmet_window_excludes_unknown_and_prunes_old_frames() -> None:
    state = make_track()
    add_helmet_observation(state, HELMET, 0.0, 3.0)
    add_helmet_observation(state, UNKNOWN, 1.0, 3.0)
    add_helmet_observation(state, ABSENT, 2.0, 3.0)

    assert state.recent_evaluable_frames == 2
    assert state.recent_helmet_ratio == 0.5
    assert state.recent_no_helmet_ratio == 0.0

    add_helmet_observation(state, NO_HELMET, 4.1, 3.0)
    assert state.recent_evaluable_frames == 2
    assert state.recent_helmet_ratio == 0.0
    assert state.recent_no_helmet_ratio == 0.5


def test_event_activates_once_and_resolves_after_recovery() -> None:
    common = {
        "event_key": "NO_HELMET:1",
        "event_type": "NO_HELMET",
        "camera_id": "camera_01",
        "zone_id": None,
        "track_id": 1,
        "metrics": {"helmet_ratio": 0.0},
        "thresholds": {"violation_helmet_ratio": 0.2},
    }

    event, transition = advance_event(
        None,
        condition=True,
        recovery_condition=False,
        now=0.0,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert event is not None
    assert transition == "PENDING"

    event, transition = advance_event(
        event,
        condition=True,
        recovery_condition=False,
        now=1.1,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert event is not None
    assert transition == "ACTIVE"
    original_event_id = event.event_id

    event, transition = advance_event(
        event,
        condition=True,
        recovery_condition=False,
        now=2.0,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert transition is None
    assert event is not None and event.event_id == original_event_id

    event, transition = advance_event(
        event,
        condition=False,
        recovery_condition=True,
        now=3.0,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert transition == "RECOVERING"
    event, transition = advance_event(
        event,
        condition=False,
        recovery_condition=True,
        now=4.1,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert event is not None
    assert transition == "RESOLVED"
    assert event.status == "RESOLVED"


def test_pending_event_is_recorded_as_cancelled_when_condition_disappears() -> None:
    common = {
        "event_key": "NO_HELMET:1",
        "event_type": "NO_HELMET",
        "camera_id": "camera_01",
        "zone_id": None,
        "track_id": 1,
        "metrics": {"helmet_ratio": 0.1},
        "thresholds": {"violation_helmet_ratio": 0.2},
    }
    event, transition = advance_event(
        None,
        condition=True,
        recovery_condition=False,
        now=0.0,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert transition == "PENDING"

    event, transition = advance_event(
        event,
        condition=False,
        recovery_condition=True,
        now=0.5,
        confirm_seconds=1.0,
        recovery_seconds=1.0,
        **common,
    )
    assert event is not None
    assert transition == "CANCELLED"
    assert event.status == "CANCELLED"
