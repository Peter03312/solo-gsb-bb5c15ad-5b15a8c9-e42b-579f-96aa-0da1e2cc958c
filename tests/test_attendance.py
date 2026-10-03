"""End-to-end tests for attendance: check-in (签到) and check-out (离场).

Check-in opens ``CHECKIN_LEAD_MINUTES`` (default 15) before the reservation
start (included) and closes at the end (excluded); check-out opens at the
start (included) and closes at the end (excluded).  Both authenticate with
the original cancel token and replay the first record on repetition.  A
check-out releases the remainder of the held slot from that instant on and
promotes the waitlist in application order inside the same transaction.
Checked-in reservations cannot be rescheduled or reassigned; their number,
original window, revision, cancel token and creation snapshot never change.
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


def parse(dt_string):
    return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))


def at(hour, minute=0, second=0, day_offset=1):
    day = (datetime.now(UTC) + timedelta(days=day_offset)).date()
    return datetime(
        day.year, day.month, day.day, hour, minute, second, tzinfo=UTC
    )


class _FrozenDateTime:
    """datetime stand-in whose now(tz) returns one fixed instant."""

    def __init__(self, frozen):
        self._frozen = frozen

    def now(self, tz=None):
        if tz is None:
            return self._frozen.replace(tzinfo=None)
        return self._frozen.astimezone(tz)


def freeze(monkeypatch, frozen):
    import app.main as main_module

    monkeypatch.setattr(main_module, "datetime", _FrozenDateTime(frozen))


def open_point(client, capacity=None, tz="UTC"):
    body = {"name": "到场测试点", "timezone": tz}
    if capacity is not None:
        body["capacity"] = capacity
    r = client.post("/points", json=body)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": [
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
    return pid


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def book_one(client, capacity=1, start=None, end=None, key="r1"):
    pid = open_point(client, capacity=capacity)
    r = reserve(client, pid, key, start or at(9), end or at(10))
    assert r.status_code == 201, r.text
    return pid, r.json()


def checkin(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/checkin",
        json={"cancel_token": token},
    )


def checkout(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/checkout",
        json={"cancel_token": token},
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


def reschedule(client, pid, rid, token, key, start, end, revision=1):
    return client.post(
        f"/points/{pid}/reservations/{rid}/reschedule",
        json={
            "cancel_token": token,
            "idempotency_key": key,
            "expected_revision": revision,
            "start": iso(start),
            "end": iso(end),
        },
    )


def reassign(client, src_pid, rid, token, key, target_pid, start, end, revision=1):
    return client.post(
        f"/points/{src_pid}/reservations/{rid}/reassign",
        json={
            "cancel_token": token,
            "idempotency_key": key,
            "expected_revision": revision,
            "target_point_id": target_pid,
            "start": iso(start),
            "end": iso(end),
        },
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


def record_attendance(rid, checkin_at, checkout_at=None):
    """Write attendance timestamps straight into the database.

    Lets the capacity tests arrange a real early departure at a fixed instant
    while booking later reservations against the ordinary (real-time) future,
    without freezing the endpoint clock for those bookings.
    """
    from app import db

    with db.conn() as connection:
        with connection.transaction():
            connection.cursor().execute(
                "UPDATE reservations SET checked_in_at = %s, "
                "checked_out_at = %s WHERE id = %s",
                (checkin_at, checkout_at, rid),
            )


# ---------------------------------------------------------------------------
# Check-in window: [start - 15m, end)
# ---------------------------------------------------------------------------


def test_checkin_exactly_15_minutes_before_start_succeeds(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 45))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["checkin_at"] is not None
    assert body["checkout_at"] is None
    assert parse(body["checkin_at"]) == at(8, 45)


def test_checkin_inside_window_succeeds(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(9, 30))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200, r.text
    assert r.json()["checkin_at"] is not None


def test_checkin_before_window_is_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 44, 59))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "attendance_window_closed"
    assert err["details"]["reason"] == "not_yet_open"

def test_checkin_at_start_succeeds(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(9))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200, r.text
    assert parse(r.json()["checkin_at"]) == at(9)


def test_checkin_at_end_is_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(10))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "attendance_window_closed"
    assert err["details"]["reason"] == "already_ended"


def test_checkin_after_end_is_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(10, 1))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    assert r.json()["error"]["details"]["reason"] == "already_ended"


# ---------------------------------------------------------------------------
# Check-in credentials and states
# ---------------------------------------------------------------------------


def test_repeat_checkin_returns_first_record(client, monkeypatch):
    import app.main as main_module

    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 45))
    first = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert first.status_code == 200
    monkeypatch.setattr(
        main_module, "datetime", _FrozenDateTime(at(9, 30))
    )
    again = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert again.status_code == 200
    assert again.json()["checkin_at"] == first.json()["checkin_at"]


def test_checkin_without_token_refused(client):
    pid, resv = book_one(client)
    r = client.post(f"/points/{pid}/reservations/{resv['id']}/checkin", json={})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_cancel_token"


def test_checkin_with_wrong_token_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(9, 30))
    r = checkin(client, pid, resv["id"], "wrong-token")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_cancel_token"


def test_checkin_unknown_reservation_is_404(client, monkeypatch):
    pid = open_point(client)
    freeze(monkeypatch, at(9, 30))
    r = checkin(client, pid, 999999, "anything")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "reservation_not_found"


def test_checkin_cancelled_reservation_refused(client, monkeypatch):
    pid, resv = book_one(client)
    assert cancel(client, pid, resv["id"], resv["cancel_token"]).status_code == 200
    freeze(monkeypatch, at(9, 30))
    r = checkin(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_attendance_conflict"
    assert err["details"]["reason"] == "reservation_cancelled"


def test_checkin_unconfirmed_lock_number_refused(client):
    pid = open_point(client)
    r = client.post(
        f"/points/{pid}/locks",
        json={"idempotency_key": "L1", "start": iso(at(9)), "end": iso(at(10))},
    )
    assert r.status_code == 201
    lock_id = r.json()["id"]
    # A number naming a still-locked lock is not a check-in-able booking.
    resp = checkin(client, pid, lock_id, "whatever-token")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "lock_not_confirmed"


def test_checkin_after_lock_expired_is_404(client, monkeypatch):
    from app import db

    pid = open_point(client)
    r = client.post(
        f"/points/{pid}/locks",
        json={"idempotency_key": "L1", "start": iso(at(9)), "end": iso(at(10))},
    )
    lock_id = r.json()["id"]
    with db.conn() as connection:
        with connection.transaction():
            connection.cursor().execute(
                "UPDATE slot_locks SET expires_at = now() - interval '1 second' "
                "WHERE id = %s",
                (lock_id,),
            )
    freeze(monkeypatch, at(9, 30))
    resp = checkin(client, pid, lock_id, "whatever-token")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "reservation_not_found"


def test_confirmed_lock_reservation_can_checkin(client, monkeypatch):
    pid = open_point(client)
    lock = client.post(
        f"/points/{pid}/locks",
        json={"idempotency_key": "L1", "start": iso(at(9)), "end": iso(at(10))},
    ).json()
    confirmed = client.post(
        f"/points/{pid}/locks/{lock['id']}/confirm",
        json={"confirm_token": lock["confirm_token"]},
    )
    assert confirmed.status_code == 200
    rid = confirmed.json()["reservation"]["id"]
    token = confirmed.json()["reservation"]["cancel_token"]
    freeze(monkeypatch, at(9, 30))
    r = checkin(client, pid, rid, token)
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Check-out window: [start, end), requires a prior check-in
# ---------------------------------------------------------------------------


def test_checkout_without_checkin_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(9, 30))
    r = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_attendance_conflict"
    assert err["details"]["reason"] == "not_checked_in"


def test_checkout_before_start_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 50))
    assert checkin(client, pid, resv["id"], resv["cancel_token"]).status_code == 200
    freeze(monkeypatch, at(8, 55))
    r = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "attendance_window_closed"
    assert err["details"]["reason"] == "not_started"


def test_checkout_inside_window_succeeds_and_releases_remainder(
    client, monkeypatch
):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 50))
    assert checkin(client, pid, resv["id"], resv["cancel_token"]).status_code == 200
    freeze(monkeypatch, at(9, 30))
    r = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert parse(body["checkout_at"]) == at(9, 30)
    # The released remainder [09:30, 10:00) is bookable again.
    follow = reserve(client, pid, "later", at(9, 30), at(10))
    assert follow.status_code == 201, follow.text


def test_checkout_at_start_succeeds(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 45))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(9))
    r = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200, r.text
    assert parse(r.json()["checkout_at"]) == at(9)


def test_checkout_at_end_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 45))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(10))
    r = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    assert r.json()["error"]["details"]["reason"] == "already_ended"


def test_repeat_checkout_returns_first_record(client, monkeypatch):
    import app.main as main_module

    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 45))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(9, 30))
    first = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert first.status_code == 200
    # Even after the window has ended, a repeat check-out replays the first.
    monkeypatch.setattr(main_module, "datetime", _FrozenDateTime(at(11)))
    again = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert again.status_code == 200
    assert again.json()["checkout_at"] == first.json()["checkout_at"]
    # The replay is credential-checked first.
    bad = checkout(client, pid, resv["id"], "wrong-token")
    assert bad.status_code == 403


def test_checkout_wrong_token_always_refused(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(9, 30))
    assert (
        checkout(client, pid, resv["id"], "wrong-token").status_code == 403
    )


def test_checkout_cancelled_after_checkin_refused_first_time(
    client, monkeypatch
):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(9, 30))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    assert cancel(client, pid, resv["id"], resv["cancel_token"]).status_code == 200
    r = checkout(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 409
    assert r.json()["error"]["details"]["reason"] == "reservation_cancelled"


# ---------------------------------------------------------------------------
# Early departure and capacity / waitlist
# ---------------------------------------------------------------------------


def test_booked_part_before_departure_still_occupies(client):
    pid, resv = book_one(client)
    record_attendance(resv["id"], at(8, 50), at(9, 30))
    # A window overlapping the consumed part [09:00, 09:30) is still full.
    assert reserve(client, pid, "overlap", at(9), at(9, 30)).status_code == 409
    assert reserve(client, pid, "overlap2", at(9, 15), at(9, 45)).status_code == 409
    # Touching the departure instant is free.
    assert reserve(client, pid, "touch", at(9, 30), at(10)).status_code == 201


def test_checkout_promotes_waitlist_in_application_order_same_tx(
    client, monkeypatch
):
    pid = open_point(client, capacity=1)
    holder = reserve(client, pid, "h", at(9), at(10)).json()
    # Two waiters for the released remainder; the earlier applicant wins.
    # Their windows start after the departure so the sweep can promote them.
    w1 = waitlist(client, pid, "w1", at(9, 30), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(9, 30), at(10)).json()
    assert w1["id"] < w2["id"]
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, holder["id"], holder["cancel_token"])
    freeze(monkeypatch, at(9, 29))
    r = checkout(client, pid, holder["id"], holder["cancel_token"])
    assert r.status_code == 200
    assert parse(r.json()["checkout_at"]) == at(9, 29)
    s1 = waitlist_status(client, pid, w1["id"], w1["query_token"]).json()
    s2 = waitlist_status(client, pid, w2["id"], w2["query_token"]).json()
    assert s1["status"] == "promoted"
    assert s2["status"] == "waiting"
    # The promoted reservation genuinely holds the freed remainder (the
    # verification booking runs against the real future clock).
    assert reserve(client, pid, "racer", at(9, 30), at(10)).status_code == 409


def test_waiter_that_does_not_fit_waits_after_early_departure(
    client, monkeypatch
):
    # Capacity 1; the holder leaves at 09:29, freeing [09:29, 10:00).  The
    # sweep scans in application order: an entry that fits is promoted and
    # keeps its slot; an entry that does not fit yet is skipped and keeps
    # waiting while later applicants are still considered.
    pid = open_point(client, capacity=1)
    holder = reserve(client, pid, "h", at(9), at(10)).json()
    w_fit = waitlist(client, pid, "w-fit", at(9, 30), at(9, 45)).json()
    w_block = waitlist(client, pid, "w-block", at(9, 30), at(10)).json()
    w_overlap = waitlist(client, pid, "w-overlap", at(9, 35), at(9, 40)).json()
    w_after = waitlist(client, pid, "w-after", at(9, 45), at(10)).json()
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, holder["id"], holder["cancel_token"])
    freeze(monkeypatch, at(9, 29))
    checkout(client, pid, holder["id"], holder["cancel_token"])
    s_fit = waitlist_status(client, pid, w_fit["id"], w_fit["query_token"]).json()
    s_block = waitlist_status(
        client, pid, w_block["id"], w_block["query_token"]
    ).json()
    s_overlap = waitlist_status(
        client, pid, w_overlap["id"], w_overlap["query_token"]
    ).json()
    s_after = waitlist_status(
        client, pid, w_after["id"], w_after["query_token"]
    ).json()
    assert s_fit["status"] == "promoted"
    assert s_block["status"] == "waiting"
    assert s_overlap["status"] == "waiting"
    # A later applicant that merely touches the promoted reservation is still
    # promoted — the non-fitting entry does not block the rest of the queue.
    assert s_after["status"] == "promoted"


def test_expired_waiter_expired_on_departure_sweep(client, monkeypatch):
    pid = open_point(client, capacity=1)
    holder = reserve(client, pid, "h", at(9), at(10)).json()
    # Applicant for a window whose start has passed by the departure sweep.
    w = waitlist(client, pid, "w", at(9), at(9, 20)).json()
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, holder["id"], holder["cancel_token"])
    freeze(monkeypatch, at(9, 30))
    checkout(client, pid, holder["id"], holder["cancel_token"])
    status = waitlist_status(client, pid, w["id"], w["query_token"]).json()
    assert status["status"] == "expired"


# ---------------------------------------------------------------------------
# Checked-in reservations cannot be moved; cancel still works
# ---------------------------------------------------------------------------


def test_checked_in_reservation_cannot_be_rescheduled(client, monkeypatch):
    pid, resv = book_one(client, start=at(9), end=at(11))
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(8, 55))
    r = reschedule(
        client, pid, resv["id"], resv["cancel_token"], "m1", at(10), at(11)
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_attendance_conflict"
    assert err["details"]["reason"] == "already_checked_in"
    # The window is untouched.
    state = cancel(client, pid, resv["id"], resv["cancel_token"]).json()
    assert state["start"] == iso(at(9))
    assert state["end"] == iso(at(11))


def test_checked_in_reservation_cannot_be_reassigned(client, monkeypatch):
    pid, resv = book_one(client, start=at(9), end=at(11))
    pid2 = open_point(client)
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(8, 55))
    r = reassign(
        client,
        pid,
        resv["id"],
        resv["cancel_token"],
        "re1",
        pid2,
        at(9),
        at(10),
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_attendance_conflict"
    assert err["details"]["reason"] == "already_checked_in"


def test_checked_in_reservation_can_still_be_cancelled(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    r = cancel(client, pid, resv["id"], resv["cancel_token"])
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


def test_reschedule_replay_still_works_after_checkin(client, monkeypatch):
    pid, resv = book_one(client, start=at(9), end=at(11))
    # A successful reschedule happened before arrival.
    moved = reschedule(
        client, pid, resv["id"], resv["cancel_token"], "m1", at(10), at(11)
    )
    assert moved.status_code == 200
    freeze(monkeypatch, at(9, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    # Replaying the old reschedule attempt returns the stored attempt instead
    # of refusing (idempotent replay wins over the checked-in guard).
    replay = reschedule(
        client, pid, resv["id"], resv["cancel_token"], "m1", at(10), at(11),
        revision=1,
    )
    assert replay.status_code == 200
    assert replay.json()["start"] == iso(at(10))


# ---------------------------------------------------------------------------
# Invariants: number / window / revision / token / creation snapshot
# ---------------------------------------------------------------------------


def test_identifiers_and_window_unchanged_through_attendance(client, monkeypatch):
    pid, resv = book_one(client)
    token = resv["cancel_token"]
    freeze(monkeypatch, at(8, 50))
    ci = checkin(client, pid, resv["id"], token).json()
    assert ci["id"] == resv["id"]
    assert ci["revision"] == 1
    assert ci["start"] == resv["start"]
    assert ci["end"] == resv["end"]
    assert ci["rules_version"] == resv["rules_version"]
    freeze(monkeypatch, at(9, 30))
    co = checkout(client, pid, resv["id"], token).json()
    assert co["id"] == resv["id"]
    assert co["revision"] == 1
    assert co["start"] == resv["start"]
    assert co["end"] == resv["end"]
    assert co["rules_version"] == resv["rules_version"]
    assert "cancel_token" not in co
    # The original creation-key retry snapshot is untouched too.
    replay = reserve(client, pid, "r1", at(9), at(10))
    assert replay.status_code == 200
    body = replay.json()
    assert body["start"] == resv["start"]
    assert body["revision"] == 1
    assert body["cancel_token"] == token


# ---------------------------------------------------------------------------
# Capacity decisions everywhere follow the effective occupancy
# ---------------------------------------------------------------------------


def test_planner_ignores_released_remainder(client, monkeypatch):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(9, 30))
    checkout(client, pid, resv["id"], resv["cancel_token"])
    # A second open point lets us search a two-stop plan starting inside the
    # released remainder; point 1 must be considered free at 09:30.
    pid2 = open_point(client)
    search_from = at(9, 30)
    search_to = at(10, 30)
    r = client.post(
        "/trips/plan",
        json={
            "stops": [
                {"point_id": pid, "duration_minutes": 30},
                {"point_id": pid2, "duration_minutes": 30},
            ],
            "min_gap_minutes": 0,
            "search_from": iso(search_from),
            "search_to": iso(search_to),
        },
    )
    assert r.status_code == 200, r.text
    stops = r.json()["stops"]
    assert stops, "expected the released remainder to admit a planned stop"
    assert parse(stops[0]["start"]) == at(9, 30)


def test_capacity_decrease_uses_effective_occupancy(client):
    # Capacity 2; r1 [09:00,10:00) and r2 [09:30,10:00) overlap originally,
    # so a decrease to 1 must be refused while both hold.  After r1 leaves at
    # exactly 09:30 the two *effective* windows merely touch (peak 1), so the
    # decrease becomes legal even though the original rows still overlap.
    pid = open_point(client, capacity=2)
    r1 = reserve(client, pid, "a", at(9), at(10)).json()
    r2 = reserve(client, pid, "b", at(9, 30), at(10)).json()
    assert client.patch(f"/points/{pid}", json={"capacity": 1}).status_code == 409
    record_attendance(r1["id"], at(9), at(9, 30))
    patch = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert patch.status_code == 200, patch.text
    assert patch.json()["capacity"] == 1
    # Both effective halves still occupy their own slots.
    assert reserve(client, pid, "c1", at(9), at(9, 30)).status_code == 409
    assert reserve(client, pid, "c2", at(9, 30), at(10)).status_code == 409
    assert reserve(client, pid, "e", at(10), at(10, 30)).status_code == 201
    # r2 keeps its original window and blocks it as before.
    assert r2["start"] == iso(at(9, 30))


def test_capacity_decrease_still_refuses_unfinished_double(client):
    # The same decrease must stay refused when both visitors are still there.
    pid = open_point(client, capacity=2)
    reserve(client, pid, "a", at(9), at(10))
    reserve(client, pid, "b", at(9, 30), at(10))
    patch = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert patch.status_code == 409
    assert patch.json()["error"]["code"] == "capacity_conflict"
    assert client.get(f"/points/{pid}").json()["capacity"] == 2


def test_trip_stop_sees_released_capacity(client, monkeypatch):
    pid, resv = book_one(client)
    pid2 = open_point(client)
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(9, 30))
    checkout(client, pid, resv["id"], resv["cancel_token"])
    trip = client.post(
        "/trips",
        json={
            "idempotency_key": "attendance-trip-release-1",
            "stops": [
                {"point_id": pid, "start": iso(at(9, 30)), "end": iso(at(10))},
                {"point_id": pid2, "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert trip.status_code == 201, trip.text


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_checkout_and_booking_never_double_occupy(
    client, monkeypatch
):
    pid = open_point(client, capacity=1)
    holder = reserve(client, pid, "h", at(9), at(10)).json()
    w = waitlist(client, pid, "w", at(9, 30), at(10)).json()
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, holder["id"], holder["cancel_token"])
    # Departure just before the waiters' window so the same-transaction sweep
    # can actually promote the waiting entry.
    freeze(monkeypatch, at(9, 29))
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_checkout():
        barrier.wait()
        outcomes["checkout"] = checkout(
            client, pid, holder["id"], holder["cancel_token"]
        ).status_code

    def do_book():
        barrier.wait()
        outcomes["book"] = reserve(
            client, pid, "racer", at(9, 30), at(10)
        ).status_code

    threads = [
        threading.Thread(target=do_checkout),
        threading.Thread(target=do_book),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes["checkout"] == 200
    entry = waitlist_status(client, pid, w["id"], w["query_token"]).json()
    # Either the same-transaction sweep promoted the waiter (racer loses) or
    # the racer booked the freed remainder first (waiter stays queued).
    if entry["status"] == "promoted":
        assert outcomes["book"] == 409
    else:
        assert outcomes["book"] == 201
        assert entry["status"] == "waiting"


def test_concurrent_repeated_checkout_records_one_departure(
    client, monkeypatch
):
    pid, resv = book_one(client)
    freeze(monkeypatch, at(8, 50))
    checkin(client, pid, resv["id"], resv["cancel_token"])
    freeze(monkeypatch, at(9, 30))
    barrier = threading.Barrier(3)
    codes = []

    def do_checkout():
        barrier.wait()
        codes.append(
            checkout(client, pid, resv["id"], resv["cancel_token"]).status_code
        )

    threads = [threading.Thread(target=do_checkout) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200, 200, 200]
    from app import db

    with db.conn() as connection:
        row = connection.cursor().execute(
            "SELECT checked_in_at, checked_out_at FROM reservations WHERE id = %s",
            (resv["id"],),
        ).fetchone()
    assert row["checked_in_at"] is not None
    assert row["checked_out_at"] is not None
