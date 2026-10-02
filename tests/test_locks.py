"""End-to-end tests for visit slot locks (到访名额锁定).

A lock holds one capacity slot of its window for the confirmation TTL
(default 10 minutes); confirming with the confirm_token before expiry
atomically converts it into an ordinary reservation.  Valid locks count as
occupancy everywhere (reservations, trips, plan search, waitlist); expired
locks are swept by the next capacity-touching write and promote the waitlist.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def at(hour, minute=0, day_offset=1):
    day = (datetime.now(UTC) + timedelta(days=day_offset)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def parse(dt_string):
    return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))


def create_point(client, capacity=None, tz="UTC"):
    body = {"name": "锁定测试点", "timezone": tz}
    if capacity is not None:
        body["capacity"] = capacity
    r = client.post("/points", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def open_point(client, capacity=None, tz="UTC", weekly=None):
    """Point with rules covering ``weekly``; default 08:00-12:00 tomorrow."""
    point = create_point(client, capacity=capacity, tz=tz)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{point['id']}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": weekly
                if weekly is not None
                else [
                    {
                        "weekday": tomorrow.isoweekday(),
                        "start_time": "08:00",
                        "end_time": "12:00",
                    }
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    return point["id"]


def lock(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/locks",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def confirm(client, pid, lid, token):
    return client.post(
        f"/points/{pid}/locks/{lid}/confirm", json={"confirm_token": token}
    )


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def waitlist(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/waitlist",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def waitlist_status(client, pid, wid, token):
    return client.get(
        f"/points/{pid}/waitlist/{wid}", params={"query_token": token}
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


def force_expire_lock(lock_id):
    """Push a lock past its confirmation deadline straight in the database."""
    from app import db

    with db.conn() as connection:
        with connection.transaction():
            connection.cursor().execute(
                "UPDATE slot_locks SET expires_at = now() - interval '1 second' "
                "WHERE id = %s",
                (lock_id,),
            )


def lock_one(client, capacity=1, start=None, end=None):
    """Open a point and take one lock on it; return (pid, lock_payload)."""
    pid = open_point(client, capacity=capacity)
    r = lock(client, pid, "L1", start or at(9), end or at(10))
    assert r.status_code == 201, r.text
    return pid, r.json()


# ---------------------------------------------------------------------------
# Creation: response shape and window policy (mirrors single reservations)
# ---------------------------------------------------------------------------


def test_create_lock_returns_hold_details(client):
    pid = open_point(client)
    r = lock(client, pid, "L1", at(9), at(10))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] > 0
    assert body["point_id"] == pid
    assert body["status"] == "locked"
    assert body["start"] == iso(at(9))
    assert body["end"] == iso(at(10))
    assert body["rules_version"] == 1
    assert body["confirm_token"]
    assert body["reservation"] is None
    # The confirmation deadline is the 10-minute TTL from creation.
    delta = parse(body["expires_at"]) - parse(body["created_at"])
    assert timedelta(minutes=9) < delta < timedelta(minutes=11)


def test_lock_window_policy_mirrors_reservation_errors(client):
    pid = open_point(client)
    now = datetime.now(UTC)
    # end not strictly after start
    r = lock(client, pid, "bad-1", at(10), at(10))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"
    # longer than the 120-minute reservation cap
    r = lock(client, pid, "bad-2", at(8), at(12))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"
    # start in the past
    r = lock(client, pid, "bad-3", now - timedelta(hours=1), now + timedelta(hours=1))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"
    # beyond the 30-day booking horizon
    r = lock(client, pid, "bad-4", at(9, day_offset=31), at(10, day_offset=31))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"
    # naive timestamps are a shape error
    r = client.post(
        f"/points/{pid}/locks",
        json={
            "idempotency_key": "bad-5",
            "start": "2026-10-01T09:00:00",
            "end": "2026-10-01T10:00:00",
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
    # outside the published open hours
    r = lock(client, pid, "bad-6", at(13), at(14))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "outside_open_hours"


def test_lock_requires_published_rules_and_existing_point(client):
    point = create_point(client)
    r = lock(client, point["id"], "L1", at(9), at(10))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "no_published_version"
    r = lock(client, 999999, "L1", at(9), at(10))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "point_not_found"


# ---------------------------------------------------------------------------
# Idempotency (per-point key scope, first successful request wins)
# ---------------------------------------------------------------------------


def test_same_key_same_content_replays_original_lock(client):
    pid, body = lock_one(client)
    replay = lock(client, pid, "L1", at(9), at(10))
    assert replay.status_code == 200
    again = replay.json()
    assert again["id"] == body["id"]
    assert again["confirm_token"] == body["confirm_token"]
    assert again["expires_at"] == body["expires_at"]
    assert again["status"] == "locked"


def test_same_key_different_content_conflicts(client):
    pid, body = lock_one(client)
    r = lock(client, pid, "L1", at(9), at(11))
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "idempotency_conflict"
    assert err["details"]["lock_id"] == body["id"]
    assert err["details"]["start"] == iso(at(9))
    assert err["details"]["end"] == iso(at(10))


def test_lock_key_scoped_per_point_and_independent_of_reservation_keys(client):
    pid1 = open_point(client)
    pid2 = open_point(client)
    assert lock(client, pid1, "shared", at(9), at(10)).status_code == 201
    # Same key on another point is an independent scope.
    assert lock(client, pid2, "shared", at(9), at(10)).status_code == 201
    # Same key string as a reservation on the same point: independent namespaces.
    r = reserve(client, pid1, "shared", at(10), at(11))
    assert r.status_code == 201, r.text


def test_replay_after_expiry_reports_expired_state(client):
    pid, body = lock_one(client)
    force_expire_lock(body["id"])
    replay = lock(client, pid, "L1", at(9), at(10))
    assert replay.status_code == 200
    assert replay.json()["id"] == body["id"]
    assert replay.json()["status"] == "expired"


# ---------------------------------------------------------------------------
# Capacity: a valid lock occupies one slot everywhere
# ---------------------------------------------------------------------------


def test_lock_occupies_capacity_against_reservations(client):
    pid, body = lock_one(client)
    # Same window: the lock already holds the only slot.
    r = reserve(client, pid, "R1", at(9), at(10))
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["conflicting_lock_ids"] == [body["id"]]
    assert err["details"]["conflicting_reservation_ids"] == []
    assert err["details"]["capacity"] == 1
    assert err["details"]["peak_concurrency"] == 1
    # Touching end-to-end is free (half-open [start, end) rule).
    assert reserve(client, pid, "R2", at(10), at(11)).status_code == 201
    assert reserve(client, pid, "R3", at(8), at(9)).status_code == 201


def test_locks_count_against_each_other(client):
    pid = open_point(client, capacity=2)
    assert lock(client, pid, "L1", at(9), at(10)).status_code == 201
    assert lock(client, pid, "L2", at(9), at(10)).status_code == 201
    r = lock(client, pid, "L3", at(9, 30), at(10, 30))
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["capacity"] == 2
    assert err["details"]["peak_concurrency"] == 2
    assert len(err["details"]["conflicting_lock_ids"]) == 2


def test_lock_counts_in_trip_booking(client):
    pid_a = open_point(client)
    pid_b = open_point(client)
    held = lock(client, pid_a, "L1", at(9), at(9, 30)).json()
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "trip-lock-1",
            "stops": [
                {"point_id": pid_a, "start": iso(at(9)), "end": iso(at(9, 30))},
                {"point_id": pid_b, "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["stop_index"] == 0
    assert err["details"]["point_id"] == pid_a
    assert err["details"]["conflicting_lock_ids"] == [held["id"]]
    # Nothing was booked for the group.
    assert reserve(client, pid_b, "check", at(10), at(10, 30)).status_code == 201


def test_lock_counts_in_reschedule_and_reassign(client):
    pid = open_point(client)
    other = open_point(client)
    held = lock(client, pid, "L1", at(10), at(11)).json()
    resv = reserve(client, pid, "R1", at(8), at(9)).json()
    # Rescheduling onto the locked window fails.
    r = client.post(
        f"/points/{pid}/reservations/{resv['id']}/reschedule",
        json={
            "cancel_token": resv["cancel_token"],
            "idempotency_key": "move-1",
            "expected_revision": 1,
            "start": iso(at(10)),
            "end": iso(at(11)),
        },
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"
    assert r.json()["error"]["details"]["conflicting_lock_ids"] == [held["id"]]
    # Reassigning onto a window locked at the target point fails.
    held2 = lock(client, other, "L2", at(9), at(10)).json()
    r = client.post(
        f"/points/{pid}/reservations/{resv['id']}/reassign",
        json={
            "cancel_token": resv["cancel_token"],
            "idempotency_key": "re-1",
            "expected_revision": 1,
            "target_point_id": other,
            "start": iso(at(9)),
            "end": iso(at(10)),
        },
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"
    assert r.json()["error"]["details"]["conflicting_lock_ids"] == [held2["id"]]
    # Both failures left state untouched: the reservation is still active.
    state = cancel(client, pid, resv["id"], resv["cancel_token"]).json()
    assert state["status"] == "cancelled"
    assert state["start"] == iso(at(8))


def test_lock_counts_in_waitlist_precondition(client):
    pid, body = lock_one(client)
    # The window is calendar-valid but full (held by the lock): waitlist applies.
    r = waitlist(client, pid, "W1", at(9), at(10))
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "waiting"
    assert r.json()["position"] == 1
    # A free window (the lock expired) is not waitlist-eligible.
    force_expire_lock(body["id"])
    r = waitlist(client, pid, "W2", at(10), at(11))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "waitlist_window_available"


def test_confirmed_lock_keeps_waitlisted_entry_waiting(client):
    pid, body = lock_one(client)
    entry = waitlist(client, pid, "W1", at(9), at(10)).json()
    # Confirming converts the hold into a reservation: still no free slot.
    confirmed = confirm(client, pid, body["id"], body["confirm_token"])
    assert confirmed.status_code == 200, confirmed.text
    status = waitlist_status(client, pid, entry["id"], entry["query_token"])
    assert status.json()["status"] == "waiting"
    # Cancelling the confirmed reservation promotes the queued entry.
    reservation = confirmed.json()["reservation"]
    r = cancel(client, pid, reservation["id"], reservation["cancel_token"])
    assert r.status_code == 200
    status = waitlist_status(client, pid, entry["id"], entry["query_token"])
    assert status.json()["status"] == "promoted"
    assert status.json()["reservation"]["start"] == iso(at(9))


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def test_confirm_converts_lock_into_ordinary_reservation(client):
    pid, body = lock_one(client)
    r = confirm(client, pid, body["id"], body["confirm_token"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] == body["id"]
    assert out["status"] == "confirmed"
    assert out["expires_at"] == body["expires_at"]  # deadline not extended
    reservation = out["reservation"]
    assert reservation["point_id"] == pid
    assert reservation["status"] == "active"
    assert reservation["revision"] == 1
    assert reservation["start"] == iso(at(9))
    assert reservation["end"] == iso(at(10))
    assert reservation["rules_version"] == 1
    assert reservation["cancel_token"]
    # The converted reservation really occupies the window.
    clash = reserve(client, pid, "R1", at(9), at(10))
    assert clash.status_code == 409
    assert clash.json()["error"]["details"]["conflicting_reservation_ids"] == [
        reservation["id"]
    ]
    # The create-key replay now reports the confirmed state with the booking.
    replay = lock(client, pid, "L1", at(9), at(10))
    assert replay.status_code == 200
    assert replay.json()["status"] == "confirmed"
    assert replay.json()["reservation"]["id"] == reservation["id"]


def test_confirm_repeat_returns_same_reservation(client):
    pid, body = lock_one(client)
    first = confirm(client, pid, body["id"], body["confirm_token"]).json()
    second = confirm(client, pid, body["id"], body["confirm_token"])
    assert second.status_code == 200
    again = second.json()
    assert again["status"] == "confirmed"
    assert again["reservation"]["id"] == first["reservation"]["id"]
    assert again["reservation"]["cancel_token"] == first["reservation"]["cancel_token"]
    assert again["expires_at"] == first["expires_at"]


def test_confirmed_reservation_is_cancellable_with_its_token(client):
    pid, body = lock_one(client)
    reservation = confirm(client, pid, body["id"], body["confirm_token"]).json()[
        "reservation"
    ]
    r = cancel(client, pid, reservation["id"], reservation["cancel_token"])
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    # The window is free again.
    assert reserve(client, pid, "R-after", at(9), at(10)).status_code == 201
    # The lock view reflects the live reservation state.
    replay = lock(client, pid, "L1", at(9), at(10)).json()
    assert replay["status"] == "confirmed"
    assert replay["reservation"]["status"] == "cancelled"


def test_confirm_token_semantics(client):
    pid, body = lock_one(client)
    # Missing token.
    r = client.post(f"/points/{pid}/locks/{body['id']}/confirm", json={})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_confirm_token"
    # Wrong token.
    r = confirm(client, pid, body["id"], "not-the-token")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_confirm_token"
    # The lock is untouched and still confirmable.
    assert confirm(client, pid, body["id"], body["confirm_token"]).status_code == 200


def test_confirm_unknown_or_foreign_lock_is_404(client):
    pid, body = lock_one(client)
    r = confirm(client, pid, 999999, body["confirm_token"])
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "lock_not_found"
    other = open_point(client)
    r = confirm(client, other, body["id"], body["confirm_token"])
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "lock_not_found"


def test_confirm_after_deadline_is_explicitly_rejected(client):
    pid, body = lock_one(client)
    force_expire_lock(body["id"])
    r = confirm(client, pid, body["id"], body["confirm_token"])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "lock_expired"
    assert err["details"]["lock_id"] == body["id"]
    # The reported deadline is past the reported confirmation attempt time.
    assert parse(err["details"]["expires_at"]) <= parse(err["details"]["now"])
    # The expired lock no longer occupies: the window is bookable again.
    assert reserve(client, pid, "R1", at(9), at(10)).status_code == 201


# ---------------------------------------------------------------------------
# Expiry: sweeps on capacity-touching writes promote the waitlist
# ---------------------------------------------------------------------------


def test_expired_lock_is_swept_and_waitlist_promoted_on_next_write(client):
    pid, body = lock_one(client)
    entry = waitlist(client, pid, "W1", at(9), at(10)).json()
    assert entry["status"] == "waiting"
    force_expire_lock(body["id"])
    # An unrelated capacity-touching write on the same point sweeps the
    # expired lock and promotes the queued application into the freed slot.
    r = reserve(client, pid, "R-other", at(11), at(12))
    assert r.status_code == 201, r.text
    status = waitlist_status(client, pid, entry["id"], entry["query_token"])
    assert status.json()["status"] == "promoted"
    assert status.json()["reservation"]["start"] == iso(at(9))
    # The promoted booking really occupies: the window is full again.
    clash = reserve(client, pid, "R-clash", at(9), at(10))
    assert clash.status_code == 409
    # The lock itself reads as expired on replay.
    replay = lock(client, pid, "L1", at(9), at(10))
    assert replay.json()["status"] == "expired"


def test_expired_lock_frees_capacity_without_any_sweep(client):
    pid, body = lock_one(client)
    force_expire_lock(body["id"])
    # Capacity queries ignore expired locks directly.
    assert reserve(client, pid, "R1", at(9), at(10)).status_code == 201


def test_cancel_in_same_transaction_as_lock_expiry_still_promotes(client):
    # A real cancellation and an expired-lock cleanup in one transaction must
    # promote into the cancelled reservation's slot as well.
    pid = open_point(client)
    resv = reserve(client, pid, "R1", at(9), at(10)).json()
    entry = waitlist(client, pid, "W1", at(9), at(10)).json()
    assert entry["status"] == "waiting"
    held = lock(client, pid, "L1", at(10), at(11)).json()
    force_expire_lock(held["id"])
    r = cancel(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200
    status = waitlist_status(client, pid, entry["id"], entry["query_token"])
    assert status.json()["status"] == "promoted"
    assert status.json()["reservation"]["start"] == iso(at(9))


def test_lock_create_sweeps_expired_locks_before_checking_capacity(client):
    pid, body = lock_one(client)
    force_expire_lock(body["id"])
    # The expired lock's slot is reclaimed before the new lock is evaluated.
    r = lock(client, pid, "L2", at(9), at(10))
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "locked"


# ---------------------------------------------------------------------------
# Change protection: calendar / timezone / capacity shrink keep locks valid
# ---------------------------------------------------------------------------


def test_rule_publish_cannot_uncover_valid_lock(client):
    pid, body = lock_one(client)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {
                        "weekday": tomorrow.isoweekday(),
                        "start_time": "14:00",
                        "end_time": "16:00",
                    }
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_change_conflict"
    assert err["details"]["conflicting_lock_ids"] == [body["id"]]
    assert err["details"]["conflicting_reservation_ids"] == []
    # Once expired, the same publish goes through.
    force_expire_lock(body["id"])
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {
                        "weekday": tomorrow.isoweekday(),
                        "start_time": "14:00",
                        "end_time": "16:00",
                    }
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text


def test_timezone_change_cannot_uncover_valid_lock(client):
    pid, body = lock_one(client)
    # 09:00-10:00 UTC is 05:00-06:00 in New York — outside 08:00-12:00 local.
    r = client.patch(f"/points/{pid}", json={"timezone": "America/New_York"})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_change_conflict"
    assert err["details"]["conflicting_lock_ids"] == [body["id"]]
    force_expire_lock(body["id"])
    r = client.patch(f"/points/{pid}", json={"timezone": "America/New_York"})
    assert r.status_code == 200, r.text
    assert r.json()["timezone"] == "America/New_York"


def test_capacity_decrease_counts_valid_locks(client):
    pid = open_point(client, capacity=3)
    assert reserve(client, pid, "R1", at(9), at(11)).status_code == 201
    held = lock(client, pid, "L1", at(10), at(12)).json()
    # Peak future occupancy is 2 (reservation + lock): lowering to 1 refused.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "capacity_conflict"
    assert err["details"]["future_peak_concurrency"] == 2
    # Lowering to the peak itself is allowed.
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 200, r.text
    assert r.json()["capacity"] == 2
    # After the lock expires, lowering to 1 succeeds.
    force_expire_lock(held["id"])
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 200, r.text
    assert r.json()["capacity"] == 1


# ---------------------------------------------------------------------------
# Trip planning: valid locks count, expired locks are ignored
# ---------------------------------------------------------------------------


def plan(client, pid_a, pid_b):
    return client.post(
        "/trips/plan",
        json={
            "stops": [
                {"point_id": pid_a, "duration_minutes": 30},
                {"point_id": pid_b, "duration_minutes": 30},
            ],
            "min_gap_minutes": 0,
            "search_from": iso(at(0)),
            "search_to": iso(at(23, 59)),
        },
    )


def test_plan_counts_valid_lock_and_ignores_expired(client):
    pid_a = open_point(client)
    pid_b = open_point(client)
    held = lock(client, pid_a, "L1", at(8), at(8, 30)).json()
    # The valid lock holds 08:00-08:30: the earliest stop moves to 08:30.
    r = plan(client, pid_a, pid_b)
    assert r.status_code == 200, r.text
    stops = r.json()["stops"]
    assert len(stops) == 2
    assert stops[0]["start"] == iso(at(8, 30))
    assert stops[1]["start"] == iso(at(9))
    # Once expired, the planner ignores the lock entirely.
    force_expire_lock(held["id"])
    r = plan(client, pid_a, pid_b)
    stops = r.json()["stops"]
    assert stops[0]["start"] == iso(at(8))
    assert stops[1]["start"] == iso(at(8, 30))
    # Planning never consumed anything: the window is still lockable/bookable.
    assert lock(client, pid_a, "L2", at(8), at(8, 30)).status_code == 201


# ---------------------------------------------------------------------------
# Concurrency: no double occupancy at the expiry instant, single conversion
# ---------------------------------------------------------------------------


def test_concurrent_confirms_create_exactly_one_reservation(client):
    pid, body = lock_one(client)
    threads_n = 8
    barrier = threading.Barrier(threads_n)
    outcomes = []

    def attempt():
        barrier.wait()
        r = confirm(client, pid, body["id"], body["confirm_token"])
        outcomes.append((r.status_code, r.json()))

    threads = [threading.Thread(target=attempt) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(status == 200 for status, _ in outcomes)
    reservation_ids = {
        payload["reservation"]["id"] for _, payload in outcomes
    }
    assert len(reservation_ids) == 1
    # Database invariant: exactly one reservation exists on that window.
    from app import db

    with db.conn() as connection:
        row = connection.cursor().execute(
            "SELECT COUNT(*) AS n FROM reservations "
            "WHERE point_id = %s AND status = 'active'",
            (pid,),
        ).fetchone()
    assert row["n"] == 1


def test_concurrent_lock_creation_same_window_admits_exactly_one(client):
    pid = open_point(client)
    threads_n = 8
    barrier = threading.Barrier(threads_n)
    outcomes = []

    def attempt(i):
        barrier.wait()
        outcomes.append(lock(client, pid, f"L-{i}", at(9), at(10)).status_code)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count(201) == 1
    assert outcomes.count(409) == threads_n - 1


def test_expiry_instant_confirm_and_promotion_never_double_occupy(client):
    pid, body = lock_one(client)
    entry = waitlist(client, pid, "W1", at(9), at(10)).json()
    force_expire_lock(body["id"])
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_confirm():
        barrier.wait()
        outcomes["confirm"] = confirm(client, pid, body["id"], body["confirm_token"])

    def do_write():
        barrier.wait()
        # A capacity-touching write: sweeps the expired lock, promotes W1.
        outcomes["write"] = reserve(client, pid, "R-other", at(11), at(12))

    threads = [
        threading.Thread(target=do_confirm),
        threading.Thread(target=do_write),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # The expired lock can never be confirmed, regardless of the interleaving.
    assert outcomes["confirm"].status_code == 409
    assert outcomes["confirm"].json()["error"]["code"] == "lock_expired"
    assert outcomes["write"].status_code == 201
    # The queued application took the freed slot exactly once.
    status = waitlist_status(client, pid, entry["id"], entry["query_token"])
    assert status.json()["status"] == "promoted"
    promoted_reservation = status.json()["reservation"]["id"]
    # And it is the sole occupant of the window.
    clash = reserve(client, pid, "R-clash", at(9), at(10))
    assert clash.status_code == 409
    assert clash.json()["error"]["details"]["conflicting_reservation_ids"] == [
        promoted_reservation
    ]
