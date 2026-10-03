"""End-to-end tests for check-in / check-out (签到/离场).

A visitor checks in with the reservation number and the original cancel
token inside [start - 15min, end); the server stamps the arrival.  A
checked-in visitor checks out inside [start, end) with the same token; the
server stamps the departure, releases the remaining occupancy from that
instant and promotes the waitlist in the same transaction.  Every capacity
decision (reservation, reschedule, reassign, trip, plan search, slot lock,
capacity change) counts the reservation over its effective window
[start, checked_out_at) once the departure is recorded.
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


def now():
    return datetime.now(UTC)


def create_point(client, capacity=None, tz="UTC"):
    body = {"name": "签到离场测试点", "timezone": tz}
    if capacity is not None:
        body["capacity"] = capacity
    r = client.post("/points", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def open_point(client, capacity=None, tz="UTC"):
    """Point whose published weekly rules cover every wall-clock minute.

    Each weekday carries a same-day 00:00-23:59 rule plus an overnight
    23:00-02:00 rule, so the merged continuous interval never breaks at
    local midnight and any now-relative test window is covered.
    """
    point = create_point(client, capacity=capacity, tz=tz)
    weekly = []
    for weekday in range(1, 8):
        weekly.append(
            {"weekday": weekday, "start_time": "00:00", "end_time": "23:59"}
        )
        weekly.append(
            {"weekday": weekday, "start_time": "23:00", "end_time": "02:00"}
        )
    r = client.post(
        f"/points/{point['id']}/rules",
        json={
            "expected_version": 0,
            "rules": {"weekly": weekly, "closed_dates": [], "makeup": []},
        },
    )
    assert r.status_code == 201, r.text
    return point["id"]


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def book(client, pid, key, start=None, end=None):
    """Create a reservation; default window [now+30min, now+90min]."""
    start = start if start is not None else now() + timedelta(minutes=30)
    end = end if end is not None else now() + timedelta(minutes=90)
    r = reserve(client, pid, key, start, end)
    assert r.status_code == 201, r.text
    return r.json()


def set_window(rid, start=None, end=None):
    """Move a reservation's live window straight in the database.

    Test-time travel: the creation snapshot columns stay untouched, exactly
    like a window that genuinely reached its start.
    """
    from app import db

    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            if start is not None:
                cur.execute(
                    "UPDATE reservations SET start_utc = %s WHERE id = %s",
                    (start, rid),
                )
            if end is not None:
                cur.execute(
                    "UPDATE reservations SET end_utc = %s WHERE id = %s",
                    (end, rid),
                )


def checkin(client, pid, rid, token="T"):
    return client.post(
        f"/points/{pid}/reservations/{rid}/checkin",
        json={"cancel_token": token},
    )


def checkout(client, pid, rid, token="T"):
    return client.post(
        f"/points/{pid}/reservations/{rid}/checkout",
        json={"cancel_token": token},
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


def reschedule(client, pid, rid, token, key, revision, start, end):
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


def reassign(client, pid, rid, token, key, revision, target, start, end):
    return client.post(
        f"/points/{pid}/reservations/{rid}/reassign",
        json={
            "cancel_token": token,
            "idempotency_key": key,
            "expected_revision": revision,
            "target_point_id": target,
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


def lock_slot(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/locks",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def make_started(client, pid, key="R1", minutes=60):
    """Book, check in and move the start into the past; return the payload.

    The visit lasts ``minutes`` (must stay within the 120-minute cap).
    """
    start = now() + timedelta(minutes=10)
    body = book(client, pid, key, start=start, end=start + timedelta(minutes=minutes))
    r = checkin(client, pid, body["id"], body["cancel_token"])
    assert r.status_code == 200, r.text
    set_window(body["id"], start=now() - timedelta(minutes=1))
    return body


# ---------------------------------------------------------------------------
# Check-in (签到)
# ---------------------------------------------------------------------------


def test_checkin_records_arrival_and_repeat_replays_first_record(client):
    pid = open_point(client)
    body = book(
        client,
        pid,
        "R1",
        start=now() + timedelta(minutes=10),
        end=now() + timedelta(minutes=70),
    )
    rid, token = body["id"], body["cancel_token"]

    before = now()
    r = checkin(client, pid, rid, token)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] == rid and out["status"] == "active"
    assert out["revision"] == 1
    assert out["start"] == body["start"] and out["end"] == body["end"]
    assert out["checked_out_at"] is None
    assert "cancel_token" not in out  # state view: no capability token
    arrival = parse(out["checked_in_at"])
    assert before - timedelta(seconds=5) <= arrival <= now() + timedelta(seconds=5)

    # A repeat check-in of the still-valid reservation replays the first
    # record instead of recording a new arrival.
    r = checkin(client, pid, rid, token)
    assert r.status_code == 200, r.text
    assert r.json()["checked_in_at"] == out["checked_in_at"]


def test_checkin_window_boundaries(client):
    # Capacity 5: the three bookings below deliberately overlap.
    pid = open_point(client, capacity=5)
    # 16 minutes before the start: too early (opens 15 minutes ahead).
    early = book(
        client,
        pid,
        "R-early",
        start=now() + timedelta(minutes=16, seconds=30),
        end=now() + timedelta(minutes=76),
    )
    r = checkin(client, pid, early["id"], early["cancel_token"])
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "checkin_not_allowed"
    assert err["details"]["reason"] == "too_early"
    assert "checkin_opens_at" in err["details"]

    # 14 minutes before the start: inside the early window.
    ok = book(
        client,
        pid,
        "R-ok",
        start=now() + timedelta(minutes=14),
        end=now() + timedelta(minutes=74),
    )
    r = checkin(client, pid, ok["id"], ok["cancel_token"])
    assert r.status_code == 200, r.text

    # At/after the end: refused.  (The too-early refusal above recorded
    # nothing either, so moving the window makes check-in succeed there.)
    set_window(early["id"], start=now() - timedelta(minutes=5))
    r = checkin(client, pid, early["id"], early["cancel_token"])
    assert r.status_code == 200, r.text
    set_window(ok["id"], end=now() - timedelta(seconds=30))
    r = checkin(client, pid, ok["id"], ok["cancel_token"])
    assert r.status_code == 200, r.text  # repeat: first record still returned
    r = checkin(client, pid, early["id"], early["cancel_token"])
    assert r.status_code == 200, r.text  # repeat as well

    late = book(
        client,
        pid,
        "R-late",
        start=now() + timedelta(minutes=10),
        end=now() + timedelta(minutes=70),
    )
    set_window(late["id"], end=now() - timedelta(seconds=30))
    r = checkin(client, pid, late["id"], late["cancel_token"])
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["reason"] == "already_ended"


def test_checkin_wrong_or_missing_token_rejected(client):
    pid = open_point(client)
    body = book(client, pid, "R1")
    r = checkin(client, pid, body["id"], "wrong-token")
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "invalid_cancel_token"
    r = client.post(f"/points/{pid}/reservations/{body['id']}/checkin", json={})
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "invalid_cancel_token"


def test_checkin_cancelled_reservation_rejected(client):
    pid = open_point(client)
    body = book(client, pid, "R1")
    rid, token = body["id"], body["cancel_token"]
    r = cancel(client, pid, rid, token)
    assert r.status_code == 200, r.text
    r = checkin(client, pid, rid, token)
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "checkin_not_allowed"
    assert err["details"]["reason"] == "reservation_cancelled"


def test_checkin_cancelled_after_checkin_still_rejected(client):
    pid = open_point(client)
    body = book(
        client,
        pid,
        "R1",
        start=now() + timedelta(minutes=10),
        end=now() + timedelta(minutes=70),
    )
    rid, token = body["id"], body["cancel_token"]
    assert checkin(client, pid, rid, token).status_code == 200
    assert cancel(client, pid, rid, token).status_code == 200
    # A cancelled reservation is refused even though a first record exists.
    r = checkin(client, pid, rid, token)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["reason"] == "reservation_cancelled"


def test_checkin_unknown_reservation_and_foreign_point(client):
    pid = open_point(client)
    other = open_point(client)
    body = book(client, pid, "R1")
    r = checkin(client, pid, 999999, "T")
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "reservation_not_found"
    # The reservation exists but belongs to another point.
    r = checkin(client, other, body["id"], body["cancel_token"])
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "reservation_not_found"


def test_checkin_and_checkout_on_unconfirmed_lock_rejected(client):
    pid = open_point(client)
    r = lock_slot(
        client,
        pid,
        "L1",
        now() + timedelta(minutes=30),
        now() + timedelta(minutes=60),
    )
    assert r.status_code == 201, r.text
    lid = r.json()["id"]
    r = checkin(client, pid, lid, r.json()["confirm_token"])
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "lock_not_confirmed"
    r = checkout(client, pid, lid, "whatever")
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "lock_not_confirmed"


def test_confirmed_lock_reservation_checks_in_normally(client):
    pid = open_point(client)
    r = lock_slot(
        client,
        pid,
        "L1",
        now() + timedelta(minutes=10),
        now() + timedelta(minutes=70),
    )
    assert r.status_code == 201, r.text
    lock = r.json()
    r = client.post(
        f"/points/{pid}/locks/{lock['id']}/confirm",
        json={"confirm_token": lock["confirm_token"]},
    )
    assert r.status_code == 200, r.text
    reservation = r.json()["reservation"]
    r = checkin(client, pid, reservation["id"], reservation["cancel_token"])
    assert r.status_code == 200, r.text
    assert r.json()["checked_in_at"] is not None


# ---------------------------------------------------------------------------
# Check-out (离场)
# ---------------------------------------------------------------------------


def test_checkout_records_departure_and_repeat_replays_first_record(client):
    pid = open_point(client)
    body = make_started(client, pid)
    rid, token = body["id"], body["cancel_token"]

    before = now()
    r = checkout(client, pid, rid, token)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] == rid and out["status"] == "active"
    assert out["revision"] == 1
    assert out["checked_in_at"] is not None
    departure = parse(out["checked_out_at"])
    assert before - timedelta(seconds=5) <= departure <= now() + timedelta(seconds=5)
    assert departure >= parse(out["checked_in_at"])
    # Window, revision and number are untouched by the departure.
    assert out["end"] == body["end"]

    r = checkout(client, pid, rid, token)
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["checked_out_at"] == out["checked_out_at"]
    assert again["checked_in_at"] == out["checked_in_at"]


def test_checkout_requires_checkin_first(client):
    pid = open_point(client)
    body = book(client, pid, "R1")
    set_window(body["id"], start=now() - timedelta(minutes=1))
    r = checkout(client, pid, body["id"], body["cancel_token"])
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "checkout_not_allowed"
    assert err["details"]["reason"] == "not_checked_in"


def test_checkout_before_start_rejected(client):
    pid = open_point(client)
    # Checked in 10 minutes early; the visit itself has not started yet.
    body = book(
        client,
        pid,
        "R1",
        start=now() + timedelta(minutes=10),
        end=now() + timedelta(minutes=70),
    )
    rid, token = body["id"], body["cancel_token"]
    assert checkin(client, pid, rid, token).status_code == 200
    r = checkout(client, pid, rid, token)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["reason"] == "too_early"


def test_checkout_after_end_rejected(client):
    pid = open_point(client)
    body = make_started(client, pid)
    set_window(body["id"], end=now() - timedelta(seconds=30))
    r = checkout(client, pid, body["id"], body["cancel_token"])
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["reason"] == "already_ended"


def test_checkout_wrong_token_always_rejected(client):
    pid = open_point(client)
    body = make_started(client, pid)
    rid, token = body["id"], body["cancel_token"]
    r = checkout(client, pid, rid, "wrong-token")
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "invalid_cancel_token"
    r = checkout(client, pid, rid, token)
    assert r.status_code == 200, r.text
    first = r.json()["checked_out_at"]
    # Even after the departure is recorded, a wrong token never replays it.
    r = checkout(client, pid, rid, "wrong-token")
    assert r.status_code == 403, r.text
    r = checkout(client, pid, rid, token)
    assert r.status_code == 200, r.text
    assert r.json()["checked_out_at"] == first


def test_checkout_after_cancel_still_records_departure(client):
    # Cancellation keeps its original token semantics and stays allowed
    # after check-in; a checked-in (then cancelled) visit inside its window
    # can still record its departure with the same token.
    pid = open_point(client)
    body = make_started(client, pid)
    rid, token = body["id"], body["cancel_token"]
    r = cancel(client, pid, rid, token)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cancelled"
    r = checkout(client, pid, rid, token)
    assert r.status_code == 200, r.text
    assert r.json()["checked_out_at"] is not None


# ---------------------------------------------------------------------------
# Interaction with reschedule / reassign / cancel
# ---------------------------------------------------------------------------


def test_checked_in_reservation_cannot_be_rescheduled_or_reassigned(client):
    pid = open_point(client)
    pid2 = open_point(client)
    body = book(
        client,
        pid,
        "R1",
        start=now() + timedelta(minutes=10),
        end=now() + timedelta(minutes=70),
    )
    rid, token = body["id"], body["cancel_token"]
    assert checkin(client, pid, rid, token).status_code == 200

    r = reschedule(
        client,
        pid,
        rid,
        token,
        "move-1",
        1,
        now() + timedelta(minutes=20),
        now() + timedelta(minutes=50),
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "reservation_not_reschedulable"
    assert err["details"]["reason"] == "reservation_checked_in"

    r = reassign(
        client,
        pid,
        rid,
        token,
        "reassign-1",
        1,
        pid2,
        now() + timedelta(minutes=20),
        now() + timedelta(minutes=50),
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "reservation_not_reschedulable"
    assert err["details"]["reason"] == "reservation_checked_in"

    # Also after the departure: a consumed visit stays unmovable.
    set_window(rid, start=now() - timedelta(minutes=1))
    assert checkout(client, pid, rid, token).status_code == 200
    r = reschedule(
        client,
        pid,
        rid,
        token,
        "move-2",
        1,
        now() + timedelta(minutes=20),
        now() + timedelta(minutes=50),
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["reason"] == "reservation_checked_in"


def test_creation_replay_snapshot_unchanged_after_attendance(client):
    pid = open_point(client)
    start = now() + timedelta(minutes=10)
    end = now() + timedelta(minutes=70)
    body = book(client, pid, "R1", start=start, end=end)
    rid, token = body["id"], body["cancel_token"]
    arrival = checkin(client, pid, rid, token).json()
    assert arrival["checked_in_at"] is not None
    set_window(rid, start=now() - timedelta(minutes=1))
    departure = checkout(client, pid, rid, token).json()
    assert departure["checked_out_at"] is not None

    # The create-key replay is byte-for-byte the pre-attendance snapshot:
    # first-created window, rules version, revision 1 and cancel token, and
    # no new fields piggy-back on the existing response shape.
    r = reserve(client, pid, "R1", start, end)
    assert r.status_code == 200, r.text
    replay = r.json()
    assert set(replay) == set(body)
    assert replay["id"] == rid
    assert replay["start"] == iso(start) and replay["end"] == iso(end)
    assert replay["revision"] == 1
    assert replay["rules_version"] == body["rules_version"]
    assert replay["cancel_token"] == token
    assert replay["status"] == "active"


# ---------------------------------------------------------------------------
# Effective occupancy after an early departure
# ---------------------------------------------------------------------------


def test_checkout_releases_tail_for_new_reservations(client):
    pid = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=60)
    rid, token = body["id"], body["cancel_token"]

    clash_start = now() + timedelta(minutes=20)
    clash_end = now() + timedelta(minutes=40)
    r = reserve(client, pid, "R2", clash_start, clash_end)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "reservation_overlap"

    assert checkout(client, pid, rid, token).status_code == 200
    # The tail is free from the departure instant: the same window books.
    r = reserve(client, pid, "R2", clash_start, clash_end)
    assert r.status_code == 201, r.text


def test_checkout_promotes_waitlist_in_application_order(client):
    pid = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=90)
    rid, token = body["id"], body["cancel_token"]

    w1_start, w1_end = now() + timedelta(minutes=20), now() + timedelta(minutes=40)
    w2_start, w2_end = now() + timedelta(minutes=30), now() + timedelta(minutes=50)
    r = waitlist(client, pid, "W1", w1_start, w1_end)
    assert r.status_code == 201, r.text
    w1 = r.json()
    assert w1["status"] == "waiting" and w1["position"] == 1
    r = waitlist(client, pid, "W2", w2_start, w2_end)
    assert r.status_code == 201, r.text
    w2 = r.json()
    assert w2["status"] == "waiting" and w2["position"] == 2

    r = checkout(client, pid, rid, token)
    assert r.status_code == 200, r.text

    # The first applicant fit the freed tail and was promoted in the same
    # transaction; the second overlaps the promoted one and keeps waiting.
    r = waitlist_status(client, pid, w1["id"], w1["query_token"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "promoted"
    assert out["reservation"]["start"] == iso(w1_start)
    assert out["reservation"]["end"] == iso(w1_end)
    assert out["reservation"]["cancel_token"]

    r = waitlist_status(client, pid, w2["id"], w2["query_token"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "waiting" and out["position"] == 1

    # Cancelling the promoted reservation frees the slot again and the
    # remaining applicant is promoted in turn.
    promoted = waitlist_status(client, pid, w1["id"], w1["query_token"]).json()
    r = cancel(
        client, pid, promoted["reservation"]["id"],
        promoted["reservation"]["cancel_token"],
    )
    assert r.status_code == 200, r.text
    r = waitlist_status(client, pid, w2["id"], w2["query_token"])
    assert r.json()["status"] == "promoted"


def test_waitlist_application_on_freed_tail_is_refused_as_available(client):
    pid = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=60)
    assert checkout(client, pid, body["id"], body["cancel_token"]).status_code == 200
    # The freed tail is bookable again, so the waitlist refuses it.
    r = waitlist(
        client,
        pid,
        "W1",
        now() + timedelta(minutes=20),
        now() + timedelta(minutes=40),
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "waitlist_window_available"


def test_reschedule_and_reassign_use_effective_occupancy(client):
    pid = open_point(client, capacity=1)
    pid2 = open_point(client, capacity=1)
    # A occupies the point but leaves early; B is a later booking.
    a = make_started(client, pid, "A", minutes=60)
    b_start, b_end = now() + timedelta(minutes=150), now() + timedelta(minutes=180)
    b = book(client, pid, "B", start=b_start, end=b_end)
    assert checkout(client, pid, a["id"], a["cancel_token"]).status_code == 200

    # B can move into A's released tail (overlap with A's booked window).
    r = reschedule(
        client,
        pid,
        b["id"],
        b["cancel_token"],
        "move-B",
        1,
        now() + timedelta(minutes=20),
        now() + timedelta(minutes=40),
    )
    assert r.status_code == 200, r.text

    # A reservation on another point can be reassigned into the freed tail.
    c = book(
        client,
        pid2,
        "C",
        start=now() + timedelta(minutes=30),
        end=now() + timedelta(minutes=50),
    )
    r = reassign(
        client,
        pid2,
        c["id"],
        c["cancel_token"],
        "reassign-C",
        1,
        pid,
        now() + timedelta(minutes=20),
        now() + timedelta(minutes=40),
    )
    assert r.status_code == 409, r.text  # B now sits there
    assert r.json()["error"]["code"] == "reservation_overlap"
    r = reassign(
        client,
        pid2,
        c["id"],
        c["cancel_token"],
        "reassign-C2",
        1,
        pid,
        now() + timedelta(minutes=45),
        now() + timedelta(minutes=55),
    )
    assert r.status_code == 200, r.text


def test_lock_creation_uses_effective_occupancy(client):
    pid = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=60)
    clash_start, clash_end = now() + timedelta(minutes=20), now() + timedelta(minutes=40)
    r = lock_slot(client, pid, "L1", clash_start, clash_end)
    assert r.status_code == 409, r.text
    assert checkout(client, pid, body["id"], body["cancel_token"]).status_code == 200
    r = lock_slot(client, pid, "L1", clash_start, clash_end)
    assert r.status_code == 201, r.text


def test_trip_booking_uses_effective_occupancy(client):
    pid = open_point(client, capacity=1)
    pid2 = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=60)
    s1, e1 = now() + timedelta(minutes=20), now() + timedelta(minutes=40)
    s2, e2 = now() + timedelta(minutes=50), now() + timedelta(minutes=70)
    trip = {
        "idempotency_key": "T1",
        "stops": [
            {"point_id": pid, "start": iso(s1), "end": iso(e1)},
            {"point_id": pid2, "start": iso(s2), "end": iso(e2)},
        ],
    }
    r = client.post("/trips", json=trip)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "reservation_overlap"
    assert r.json()["error"]["details"]["stop_index"] == 0

    assert checkout(client, pid, body["id"], body["cancel_token"]).status_code == 200
    r = client.post("/trips", json=trip)
    assert r.status_code == 201, r.text
    assert [s["point_id"] for s in r.json()["stops"]] == [pid, pid2]


def test_plan_search_uses_effective_occupancy(client):
    pid = open_point(client, capacity=1)
    pid2 = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=90)
    booked_end = parse(body["end"])

    def plan():
        return client.post(
            "/trips/plan",
            json={
                "stops": [
                    {"point_id": pid, "duration_minutes": 30},
                    {"point_id": pid2, "duration_minutes": 30},
                ],
                "min_gap_minutes": 0,
                "search_from": iso(now()),
                "search_to": iso(now() + timedelta(hours=12)),
            },
        )

    r = plan()
    assert r.status_code == 200, r.text
    blocked = r.json()["stops"]
    assert len(blocked) == 2
    # Capacity 1 and the visit occupies its window: the first stop can only
    # start once the booked window ends.
    assert parse(blocked[0]["start"]) >= booked_end

    assert checkout(client, pid, body["id"], body["cancel_token"]).status_code == 200
    r = plan()
    assert r.status_code == 200, r.text
    freed = r.json()["stops"]
    assert len(freed) == 2
    # After the early departure the tail is free: the stop moves earlier
    # than the booked end.
    assert parse(freed[0]["start"]) < booked_end


def test_capacity_decrease_uses_effective_occupancy(client):
    pid = open_point(client, capacity=2)
    a = make_started(client, pid, "A", minutes=60)
    b = book(
        client,
        pid,
        "B",
        start=now() + timedelta(minutes=10),
        end=now() + timedelta(minutes=60),
    )
    # Two concurrent visits (A ongoing, B upcoming): lowering to 1 conflicts.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "capacity_conflict"

    assert checkout(client, pid, a["id"], a["cancel_token"]).status_code == 200
    # A's remaining occupancy is released: only B counts now.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 200, r.text
    assert r.json()["capacity"] == 1
    # And B can still be cancelled with its own token afterwards.
    assert cancel(client, pid, b["id"], b["cancel_token"]).status_code == 200


# ---------------------------------------------------------------------------
# Concurrency: departure vs allocation never double-books
# ---------------------------------------------------------------------------


def test_concurrent_checkout_and_booking_serialize(client):
    pid = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=60)
    rid, token = body["id"], body["cancel_token"]
    clash_start, clash_end = now() + timedelta(minutes=20), now() + timedelta(minutes=40)

    results = {}

    def do_checkout():
        results["checkout"] = checkout(client, pid, rid, token).status_code

    def do_book():
        results["book"] = reserve(client, pid, "R2", clash_start, clash_end).status_code

    threads = [threading.Thread(target=do_checkout), threading.Thread(target=do_book)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The departure always lands; the concurrent booking either squeezed
    # into the freed tail (checkout committed first) or was refused while
    # the visit still occupied its window.  Never a 500, never both
    # occupying the same instant afterwards.
    assert results["checkout"] == 200
    assert results["book"] in (201, 409)
    if results["book"] == 409:
        # The slot is free now, so a retry succeeds.
        assert reserve(client, pid, "R2", clash_start, clash_end).status_code == 201


def test_concurrent_double_checkout_records_once(client):
    pid = open_point(client, capacity=1)
    body = make_started(client, pid, minutes=60)
    rid, token = body["id"], body["cancel_token"]

    outcomes = []

    def do_checkout():
        r = checkout(client, pid, rid, token)
        outcomes.append((r.status_code, r.json()))

    threads = [threading.Thread(target=do_checkout) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [code for code, _ in outcomes] == [200, 200]
    stamps = {payload["checked_out_at"] for _, payload in outcomes}
    assert len(stamps) == 1  # both callers see the same first record
