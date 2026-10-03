"""FastAPI application: recycling-point service calendar HTTP API."""

from __future__ import annotations

import secrets
from datetime import date, datetime, timedelta, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import psycopg

from . import db
from .config import settings
from .errors import ApiError
from .errors import (
    CapacityConflictError,
    CheckinNotAllowedError,
    CheckoutNotAllowedError,
    IdempotencyConflictError,
    InvalidPeriodError,
    InvalidRequestError,
    InvalidWaitlistTokenError,
    LockExpiredError,
    LockNotConfirmedError,
    NotFoundError,
    OutsideOpenHoursError,
    ReservationNotFoundError,
    ReservationNotReschedulableError,
    ReservationOverlapError,
    RevisionConflictError,
    WaitlistWindowAvailableError,
)
from .schemas import (
    CalendarOut,
    DayOut,
    IntervalOut,
    LockConfirmIn,
    LockCreateIn,
    LockOut,
    PointCreateIn,
    PointOut,
    PointUpdateIn,
    PublishOut,
    PublishRulesIn,
    ReassignOut,
    ReservationAttendanceOut,
    ReservationCancelIn,
    ReservationCheckinIn,
    ReservationCheckoutIn,
    ReservationCreateIn,
    ReservationOut,
    ReservationReassignIn,
    ReservationRescheduleIn,
    ReservationStateOut,
    TripCreateIn,
    TripOut,
    TripPlanIn,
    TripPlanOut,
    TripPlanStopOut,
    TripStopOut,
    VersionOut,
    VersionSummaryOut,
    WaitlistCreateIn,
    WaitlistOut,
    WaitlistTokenIn,
    utc_iso,
    validate_reservation_window,
    validate_rules_payload,
    validate_timezone,
)
from .scheduler import build_calendar, build_open_intervals, get_zone
from .planner import StopSpec, plan_itinerary
from .db import payload_to_ruleset


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db(settings.database_url)
    yield
    db.close_pool()


app = FastAPI(
    title="Recycling Point Service Calendar API",
    version="1.0.0",
    description=(
        "Versioned weekly openings, full-day closures and same-day makeup "
        "openings for community recycling points, expanded to UTC intervals."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Error handling — every error is a distinguishable JSON envelope
# ---------------------------------------------------------------------------

@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "request payload failed validation",
                # Validator exceptions live in the error context; render them
                # as text so the envelope is always JSON-serializable.
                "details": {
                    "errors": jsonable_encoder(
                        exc.errors(), custom_encoder={Exception: str}
                    )
                },
            }
        },
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

@app.post("/points", response_model=PointOut, status_code=201, tags=["points"])
def create_point(body: PointCreateIn) -> PointOut:
    validate_timezone(body.timezone)
    with db.conn() as connection:
        with connection.transaction():
            row = db.create_point(
                connection.cursor(), body.name, body.timezone, body.capacity
            )
    return PointOut(**row)


@app.get("/points/{point_id}", response_model=PointOut, tags=["points"])
def read_point(point_id: int) -> PointOut:
    with db.conn() as connection:
        row = db.get_point(connection.cursor(), point_id)
    return PointOut(**row)


@app.patch("/points/{point_id}", response_model=PointOut, tags=["points"])
def patch_point(point_id: int, body: PointUpdateIn) -> PointOut:
    fields = body.model_dump(exclude_unset=True)
    if "timezone" in fields:
        validate_timezone(fields["timezone"])
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            capacity_increase = False
            if "timezone" in fields or "capacity" in fields:
                # A timezone change re-interprets every wall-clock rule; it is
                # refused when it would un-cover an active reservation or a
                # valid lock.  A capacity decrease is refused when any future
                # instant is already occupied (reservations plus valid locks)
                # at least as full as the new capacity.  The point row lock
                # serializes both checks against concurrent reservations and
                # locks; it also serializes the waitlist sweep that a
                # capacity increase triggers below.
                point = db.lock_point(cur, point_id)
                # Capacity-touching write: expire due locks first and promote
                # the waitlist into the freed slots, in application order.
                _sweep_expired_locks(cur, point, datetime.now(timezone.utc))
                if "timezone" in fields:
                    latest = db.get_latest_payload(cur, point_id)
                    if latest is not None:
                        db.assert_reservations_fit(
                            cur, point_id, fields["timezone"], latest["payload"]
                        )
                if "capacity" in fields:
                    new_capacity = fields["capacity"]
                    if new_capacity < point["capacity"]:
                        peak = db.future_peak_concurrency(
                            cur, point_id, datetime.now(timezone.utc)
                        )
                        if peak > new_capacity:
                            raise CapacityConflictError(
                                f"capacity of point {point_id} cannot be lowered "
                                f"to {new_capacity}: {peak} non-cancelled "
                                f"reservation(s) already coincide at some future "
                                f"instant",
                                details={
                                    "point_id": point_id,
                                    "requested_capacity": new_capacity,
                                    "current_capacity": point["capacity"],
                                    "future_peak_concurrency": peak,
                                },
                            )
                    capacity_increase = new_capacity > point["capacity"]
            row = db.update_point(cur, point_id, **fields)
            if capacity_increase:
                # Newly opened capacity: in the same transaction, promote the
                # earliest waitlist entries that now fit.  If the sweep
                # fails, the capacity change rolls back together with it.
                latest = db.get_latest_payload(cur, point_id)
                if latest is not None:
                    point["capacity"] = row["capacity"]
                    if "timezone" in fields:
                        point["timezone"] = row["timezone"]
                    db.promote_waitlist(
                        cur, point, latest, datetime.now(timezone.utc)
                    )
    return PointOut(**row)


# ---------------------------------------------------------------------------
# Versioned rule sets
# ---------------------------------------------------------------------------

@app.post(
    "/points/{point_id}/rules",
    response_model=PublishOut,
    status_code=201,
    tags=["rules"],
)
def publish_rules(point_id: int, body: PublishRulesIn) -> PublishOut:
    # Semantic validation (invalid periods, duplicates) before touching the DB.
    payload = validate_rules_payload(body.rules)
    with db.conn() as connection:
        with connection.transaction():
            row = db.publish_rules(
                connection.cursor(), point_id, body.expected_version, payload
            )
    return PublishOut(
        point_id=point_id, version=row["version"], published_at=row["published_at"]
    )


@app.get(
    "/points/{point_id}/rules",
    response_model=VersionOut,
    tags=["rules"],
)
def read_current_rules(point_id: int) -> VersionOut:
    with db.conn() as connection:
        point, row = db.get_current_rules(connection.cursor(), point_id)
    return VersionOut(
        point_id=point_id,
        version=row["version"],
        published_at=row["published_at"],
        rules=row["payload"],
    )


@app.get(
    "/points/{point_id}/rules/versions",
    response_model=list[VersionSummaryOut],
    tags=["rules"],
)
def list_rule_versions(point_id: int) -> list[VersionSummaryOut]:
    with db.conn() as connection:
        rows = db.list_versions(connection.cursor(), point_id)
    return [VersionSummaryOut(**row) for row in rows]


@app.get(
    "/points/{point_id}/rules/versions/{version}",
    response_model=VersionOut,
    tags=["rules"],
)
def read_rule_version(point_id: int, version: int) -> VersionOut:
    with db.conn() as connection:
        row = db.get_version(connection.cursor(), point_id, version)
    return VersionOut(
        point_id=point_id,
        version=row["version"],
        published_at=row["published_at"],
        rules=row["payload"],
    )


# ---------------------------------------------------------------------------
# Calendar query — always against one published version
# ---------------------------------------------------------------------------

@app.get(
    "/points/{point_id}/calendar",
    response_model=CalendarOut,
    tags=["calendar"],
)
def calendar(
    point_id: int,
    date_from: date = Query(..., alias="from"),
    date_to: date = Query(..., alias="to"),
    version: int | None = Query(
        default=None,
        ge=1,
        description="Pin the query to one published version; defaults to latest.",
    ),
) -> CalendarOut:
    if date_to < date_from:
        raise InvalidPeriodError("'to' must be on or after 'from'")
    span = (date_to - date_from).days + 1
    if span > settings.max_query_days:
        raise InvalidPeriodError(
            f"date range spans {span} days; maximum is {settings.max_query_days}",
            details={"max_query_days": settings.max_query_days, "span_days": span},
        )
    with db.conn() as connection:
        # Repeatable-read snapshot: the point row and the loaded rule version
        # always agree, even if a concurrent publish commits mid-query.
        previous_level = connection.isolation_level
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        try:
            with connection.transaction():
                cur = connection.cursor()
                if version is None:
                    point, row = db.get_current_rules(cur, point_id)
                else:
                    point, row = db.get_pinned_rules(cur, point_id, version)
        finally:
            connection.isolation_level = previous_level
    used_version = row["version"]
    ruleset = payload_to_ruleset(row["payload"])
    days = build_calendar(point["timezone"], ruleset, date_from, date_to)
    return CalendarOut(
        point_id=point_id,
        version=used_version,
        timezone=point["timezone"],
        date_from=date_from,
        date_to=date_to,
        days=[
            DayOut(
                local_date=d.local_date,
                intervals=[
                    IntervalOut(
                        start=utc_iso(start),
                        end=utc_iso(end),
                        sources=[s.as_dict() for s in sources],
                    )
                    for start, end, sources in d.intervals
                ],
            )
            for d in days
        ],
    )


# ---------------------------------------------------------------------------
# Reservations — booked against the currently published calendar
# ---------------------------------------------------------------------------

def _reservation_out(row: dict) -> ReservationOut:
    return ReservationOut(
        id=row["id"],
        point_id=row["point_id"],
        status=row["status"],
        revision=row["revision"],
        start=utc_iso(row["start_utc"]),
        end=utc_iso(row["end_utc"]),
        rules_version=row["rules_version"],
        cancel_token=row["cancel_token"],
        created_at=row["created_at"],
        cancelled_at=row["cancelled_at"],
    )


def _creation_replay_out(row: dict) -> ReservationOut:
    """First-creation snapshot, with the reservation's live status.

    A create-key replay always reports the *original* created window, rules
    version and revision 1 (even after reschedules), while status /
    cancelled_at reflect the reservation as it stands now.
    """
    return ReservationOut(
        id=row["id"],
        point_id=row["point_id"],
        status=row["status"],
        revision=1,
        start=utc_iso(row["created_start_utc"]),
        end=utc_iso(row["created_end_utc"]),
        rules_version=row["created_rules_version"],
        cancel_token=row["cancel_token"],
        created_at=row["created_at"],
        cancelled_at=row["cancelled_at"],
    )


def _reservation_state_out(row: dict) -> ReservationStateOut:
    return ReservationStateOut(
        id=row["id"],
        point_id=row["point_id"],
        status=row["status"],
        revision=row["revision"],
        start=utc_iso(row["start_utc"]),
        end=utc_iso(row["end_utc"]),
        rules_version=row["rules_version"],
        created_at=row["created_at"],
        cancelled_at=row["cancelled_at"],
    )


def _reservation_attendance_out(row: dict) -> ReservationAttendanceOut:
    """Check-in / check-out result: the public state plus the attendance."""
    return ReservationAttendanceOut(
        id=row["id"],
        point_id=row["point_id"],
        status=row["status"],
        revision=row["revision"],
        start=utc_iso(row["start_utc"]),
        end=utc_iso(row["end_utc"]),
        rules_version=row["rules_version"],
        created_at=row["created_at"],
        cancelled_at=row["cancelled_at"],
        checked_in_at=row["checked_in_at"],
        checked_out_at=row["checked_out_at"],
    )


def _assert_window_has_capacity(
    cur,
    point: dict,
    start: datetime,
    end: datetime,
    *,
    now: datetime,
    exclude_reservation_id: int | None = None,
) -> None:
    """Reject a window unless every instant of ``[start, end)`` is below cap.

    Occupancy counts non-cancelled reservations and valid (unexpired,
    unconfirmed) slot locks, one slot each, by the half-open interval rule:
    touching end-to-end is free and a window starting exactly at ``start``
    is already counted.  The refusal keeps the existing
    ``reservation_overlap`` error code and its
    ``conflicting_reservation_ids`` detail, extended with the conflicting
    lock ids, the point's capacity and the window's peak.
    """
    peak = db.window_peak_concurrency(
        cur,
        point["id"],
        start,
        end,
        now=now,
        exclude_reservation_id=exclude_reservation_id,
    )
    if peak >= point["capacity"]:
        overlapping = db.find_overlapping(
            cur,
            point["id"],
            start,
            end,
            exclude_reservation_id=exclude_reservation_id,
        )
        overlapping_locks = db.find_overlapping_locks(
            cur, point["id"], start, end, now
        )
        raise ReservationOverlapError(
            "reservation window is at the point's capacity for some instant",
            details={
                "conflicting_reservation_ids": [r["id"] for r in overlapping],
                "conflicting_lock_ids": [l["id"] for l in overlapping_locks],
                "capacity": point["capacity"],
                "peak_concurrency": peak,
            },
        )


def _sweep_expired_locks(cur, point: dict, now: datetime) -> bool:
    """Expire this point's due locks and promote the waitlist into freed slots.

    Every capacity-touching write runs this first (while holding the point
    row lock): an expired lock stops occupying its slot, and queued waitlist
    applications take the freed slots in application order before the write
    itself proceeds.  Returns True when at least one lock expired.
    """
    if not db.expire_locks(cur, point["id"], now):
        return False
    # A point with expirable locks necessarily has a published version (lock
    # creation requires one, and published versions are never removed).
    _, rules_row = db.get_current_rules(cur, point["id"])
    db.promote_waitlist(cur, point, rules_row, now)
    return True


@app.post(
    "/points/{point_id}/reservations",
    response_model=ReservationOut,
    status_code=201,
    tags=["reservations"],
)
def create_reservation(
    point_id: int, body: ReservationCreateIn, response: Response
) -> ReservationOut:
    start = body.start.astimezone(timezone.utc)
    end = body.end.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # The point row lock serializes creations against each other and
            # against rule publishes / timezone changes for this point.
            point = db.lock_point(cur, point_id)

            # Idempotency: the first successful request wins.  A same-content
            # retry replays the *first creation* snapshot (even if it has since
            # been rescheduled or cancelled); different content under the same
            # key conflicts.
            existing = db.get_reservation_by_key(
                cur, point_id, body.idempotency_key
            )
            if existing is not None:
                if (
                    existing["created_start_utc"] == start
                    and existing["created_end_utc"] == end
                ):
                    response.status_code = 200
                    return _creation_replay_out(existing)
                raise IdempotencyConflictError(
                    f"idempotency key {body.idempotency_key!r} was already used "
                    f"on point {point_id} with a different time window",
                    details={
                        "reservation_id": existing["id"],
                        "start": utc_iso(existing["created_start_utc"]),
                        "end": utc_iso(existing["created_end_utc"]),
                    },
                )

            validate_reservation_window(
                start,
                end,
                now,
                max_days_ahead=settings.reservation_max_days_ahead,
                max_duration=timedelta(
                    minutes=settings.reservation_max_duration_minutes
                ),
            )

            # Capacity-touching write: expire due locks first and promote the
            # waitlist into the freed slots, in application order.
            _sweep_expired_locks(cur, point, now)

            # The reservation must fit inside one continuous open interval of
            # the currently published calendar (intervals may cross local
            # midnight; they are not clipped per day here).
            _, rules_row = db.get_current_rules(cur, point_id)
            zone = get_zone(point["timezone"])
            ruleset = payload_to_ruleset(rules_row["payload"])
            date_from = (start - timedelta(days=1)).astimezone(zone).date()
            date_to = (end + timedelta(days=1)).astimezone(zone).date()
            intervals = build_open_intervals(
                point["timezone"], ruleset, date_from, date_to
            )
            if not any(s <= start and end <= e for s, e in intervals):
                raise OutsideOpenHoursError(
                    "reservation is not fully contained in one continuous "
                    "open interval of the published calendar",
                    details={
                        "start": utc_iso(start),
                        "end": utc_iso(end),
                        "rules_version": rules_row["version"],
                    },
                )

            _assert_window_has_capacity(cur, point, start, end, now=now)

            row = db.insert_reservation(
                cur,
                point_id,
                body.idempotency_key,
                start,
                end,
                rules_row["version"],
            )
    return _reservation_out(row)


@app.post(
    "/points/{point_id}/reservations/{reservation_id}/cancel",
    response_model=ReservationStateOut,
    tags=["reservations"],
)
def cancel_reservation(
    point_id: int, reservation_id: int, body: ReservationCancelIn
) -> ReservationStateOut:
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Take the point lock first (same order as every other mutating
            # path) so the subsequent waitlist sweep is race-free against
            # concurrent booking, rescheduling and rule publication.
            point = db.lock_point(cur, point_id)
            # Capacity-touching write: expire due locks first; each expiry
            # frees its held slot and promotes fitting waitlist entries.
            _sweep_expired_locks(cur, point, now)
            row, released = db.cancel_reservation(
                cur, point_id, reservation_id, body.cancel_token
            )
            if released:
                # A slot actually opened: promote, in application order, every
                # waitlist entry that now fits, all in the same
                # transaction.  If a promotion fails, the cancellation is
                # rolled back together with the sweep.  (This pass also
                # covers anything the lock sweep above could not yet see:
                # the slot freed by this very cancellation.)
                _, rules_row = db.get_current_rules(cur, point_id)
                db.promote_waitlist(cur, point, rules_row, now)
    return _reservation_state_out(row)


# ---------------------------------------------------------------------------
# Check-in / check-out (签到/离场) — attendance against the booked window
# ---------------------------------------------------------------------------

def _reservation_or_lock_refusal(
    cur, point_id: int, reservation_id: int
) -> None:
    """Raise the distinguishable refusal for an unknown reservation number.

    A number that names a slot lock which was never confirmed into a
    reservation is refused with ``409 lock_not_confirmed`` (a lock holds no
    visit yet, so there is nothing to check in or out of); anything else is
    the ordinary ``404 reservation_not_found``.
    """
    lock = db.find_lock_by_id(cur, point_id, reservation_id)
    if lock is not None and lock["reservation_id"] is None:
        raise LockNotConfirmedError(
            f"number {reservation_id} of point {point_id} is a slot lock that "
            f"has not been confirmed into a reservation; confirm the lock "
            f"first — check-in/check-out apply to reservations",
            details={
                "point_id": point_id,
                "lock_id": lock["id"],
                "lock_status": lock["status"],
            },
        )
    raise ReservationNotFoundError(
        f"reservation {reservation_id} of point {point_id} does not exist",
        details={"point_id": point_id, "reservation_id": reservation_id},
    )


@app.post(
    "/points/{point_id}/reservations/{reservation_id}/checkin",
    response_model=ReservationAttendanceOut,
    tags=["reservations"],
)
def checkin_reservation(
    point_id: int, reservation_id: int, body: ReservationCheckinIn
) -> ReservationAttendanceOut:
    """Record the visitor's arrival (签到) against a valid reservation.

    Check-in is accepted inside ``[start - CHECKIN_EARLY_MINUTES, end)``
    (15 minutes before the start, inclusive, up to the end, exclusive, by
    default) and stamps the server-side arrival time.  A repeat check-in of
    a still-valid reservation replays the first record; a wrong token, a
    cancelled reservation or an unconfirmed slot lock is refused.
    """
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Same per-point lock order as every other mutation, so the
            # attendance record serializes with cancels and reschedules.
            db.lock_point(cur, point_id)
            resv = db.find_reservation_for_update(cur, point_id, reservation_id)
            if resv is None:
                _reservation_or_lock_refusal(cur, point_id, reservation_id)
            # 404 / 403 semantics mirror the cancel endpoint: existence
            # first, then the capability token.
            db.check_cancel_token(resv, body.cancel_token)
            if resv["status"] == "cancelled":
                raise CheckinNotAllowedError(
                    f"reservation {reservation_id} is cancelled and cannot be "
                    f"checked in",
                    details={
                        "reservation_id": reservation_id,
                        "status": "cancelled",
                        "reason": "reservation_cancelled",
                    },
                )
            if resv["checked_in_at"] is not None:
                # Repeat check-in of a valid reservation: replay the first
                # recorded arrival (the record is never overwritten).
                return _reservation_attendance_out(resv)
            opens_at = resv["start_utc"] - timedelta(
                minutes=settings.checkin_early_minutes
            )
            if now < opens_at:
                raise CheckinNotAllowedError(
                    f"check-in for reservation {reservation_id} opens at "
                    f"{utc_iso(opens_at)} "
                    f"({settings.checkin_early_minutes} minutes before its "
                    f"start)",
                    details={
                        "reservation_id": reservation_id,
                        "reason": "too_early",
                        "checkin_opens_at": utc_iso(opens_at),
                        "start": utc_iso(resv["start_utc"]),
                        "now": utc_iso(now),
                    },
                )
            if now >= resv["end_utc"]:
                raise CheckinNotAllowedError(
                    f"reservation {reservation_id} ended at "
                    f"{utc_iso(resv['end_utc'])} and can no longer be "
                    f"checked in",
                    details={
                        "reservation_id": reservation_id,
                        "reason": "already_ended",
                        "end": utc_iso(resv["end_utc"]),
                        "now": utc_iso(now),
                    },
                )
            row = db.record_checkin(cur, reservation_id, now)
            return _reservation_attendance_out(row)


@app.post(
    "/points/{point_id}/reservations/{reservation_id}/checkout",
    response_model=ReservationAttendanceOut,
    tags=["reservations"],
)
def checkout_reservation(
    point_id: int, reservation_id: int, body: ReservationCheckoutIn
) -> ReservationAttendanceOut:
    """Record the visitor's departure (离场) and free the rest of the window.

    A checked-in visitor may check out inside ``[start, end)`` with the same
    capability token; the server stamps the departure time and releases the
    remaining occupancy from that instant, promoting fitting waitlist
    entries in the same transaction.  A repeat check-out replays the first
    record; a wrong token is always refused, and a first check-out without
    a check-in or after the window's end is refused.
    """
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Same per-point lock as every other capacity decision: the
            # departure record, the occupancy release and the waitlist
            # promotion serialize against concurrent bookings, so a
            # concurrent departure and allocation can never double-book.
            point = db.lock_point(cur, point_id)
            resv = db.find_reservation_for_update(cur, point_id, reservation_id)
            if resv is None:
                _reservation_or_lock_refusal(cur, point_id, reservation_id)
            # A wrong token is always refused — even when the departure was
            # already recorded.
            db.check_cancel_token(resv, body.cancel_token)
            if resv["checked_out_at"] is not None:
                # Repeat check-out with the valid token: replay the first
                # recorded departure (the record is never overwritten).
                return _reservation_attendance_out(resv)
            if resv["checked_in_at"] is None:
                raise CheckoutNotAllowedError(
                    f"reservation {reservation_id} has not checked in and "
                    f"cannot be checked out",
                    details={
                        "reservation_id": reservation_id,
                        "reason": "not_checked_in",
                    },
                )
            if now < resv["start_utc"]:
                raise CheckoutNotAllowedError(
                    f"reservation {reservation_id} starts at "
                    f"{utc_iso(resv['start_utc'])}; check-out is possible "
                    f"from the start of the visit on",
                    details={
                        "reservation_id": reservation_id,
                        "reason": "too_early",
                        "start": utc_iso(resv["start_utc"]),
                        "now": utc_iso(now),
                    },
                )
            if now >= resv["end_utc"]:
                raise CheckoutNotAllowedError(
                    f"reservation {reservation_id} ended at "
                    f"{utc_iso(resv['end_utc'])} and can no longer be "
                    f"checked out",
                    details={
                        "reservation_id": reservation_id,
                        "reason": "already_ended",
                        "end": utc_iso(resv["end_utc"]),
                        "now": utc_iso(now),
                    },
                )
            # Capacity-touching write: expire due locks first and promote the
            # waitlist into the freed slots, in application order.
            _sweep_expired_locks(cur, point, now)
            row = db.record_checkout(cur, reservation_id, now)
            # The departure frees the rest of the booked window from this
            # instant on: in the same transaction, promote every waitlist
            # entry that now fits, in application order.  A failure here
            # rolls the check-out back together with the sweep.
            _, rules_row = db.get_current_rules(cur, point_id)
            db.promote_waitlist(cur, point, rules_row, now)
            return _reservation_attendance_out(row)


def _reschedule_replay_out(reservation: dict, attempt: dict) -> ReservationOut:
    """Build the response of a successful reschedule attempt on replay.

    The window / rules version / revision come from the stored attempt; the
    status fields come from the reservation's current live row (so a replay
    after a later cancellation reports ``cancelled``).
    """
    return ReservationOut(
        id=reservation["id"],
        point_id=reservation["point_id"],
        status=reservation["status"],
        revision=attempt["revision"],
        start=utc_iso(attempt["start_utc"]),
        end=utc_iso(attempt["end_utc"]),
        rules_version=attempt["rules_version"],
        cancel_token=reservation["cancel_token"],
        created_at=reservation["created_at"],
        cancelled_at=reservation["cancelled_at"],
    )


@app.post(
    "/points/{point_id}/reservations/{reservation_id}/reschedule",
    response_model=ReservationOut,
    tags=["reservations"],
)
def reschedule_reservation(
    point_id: int, reservation_id: int, body: ReservationRescheduleIn
) -> ReservationOut:
    start = body.start.astimezone(timezone.utc)
    end = body.end.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Same per-point lock as creation / rule publishing / timezone
            # changes, so a reschedule never races a calendar change or
            # another booking, and calendar coverage + non-overlap hold under
            # concurrency.
            point = db.lock_point(cur, point_id)

            # 404 / 403 semantics mirror the cancel endpoint: existence first,
            # then the capability token.
            resv = db.get_reservation_for_update(cur, point_id, reservation_id)
            db.check_cancel_token(resv, body.cancel_token)

            # Reschedule idempotency is scoped to one reservation: same key +
            # same content replays that successful attempt (even when the
            # reservation has since moved on or been cancelled); same key +
            # different content conflicts.  Replay wins over every other
            # state/revision check.
            attempt = db.get_reschedule_by_key(
                cur, reservation_id, body.idempotency_key
            )
            if attempt is not None:
                if attempt["start_utc"] == start and attempt["end_utc"] == end:
                    return _reschedule_replay_out(resv, attempt)
                raise IdempotencyConflictError(
                    f"reschedule idempotency key {body.idempotency_key!r} was "
                    f"already used on reservation {reservation_id} with a "
                    f"different time window",
                    details={
                        "reservation_id": reservation_id,
                        "start": utc_iso(attempt["start_utc"]),
                        "end": utc_iso(attempt["end_utc"]),
                    },
                )

            if resv["status"] == "cancelled":
                raise ReservationNotReschedulableError(
                    f"reservation {reservation_id} is cancelled and cannot be "
                    f"rescheduled",
                    details={
                        "reservation_id": reservation_id,
                        "status": "cancelled",
                        "reason": "reservation_cancelled",
                    },
                )
            if resv["checked_in_at"] is not None:
                # A checked-in visit is being (or has been) consumed: its
                # window can no longer be moved, even before the start.
                raise ReservationNotReschedulableError(
                    f"reservation {reservation_id} has checked in and cannot "
                    f"be rescheduled",
                    details={
                        "reservation_id": reservation_id,
                        "status": resv["status"],
                        "reason": "reservation_checked_in",
                        "checked_in_at": utc_iso(resv["checked_in_at"]),
                    },
                )
            if resv["start_utc"] <= now:
                raise ReservationNotReschedulableError(
                    f"reservation {reservation_id} has already started and "
                    f"cannot be rescheduled",
                    details={
                        "reservation_id": reservation_id,
                        "status": resv["status"],
                        "reason": "reservation_started",
                        "start": utc_iso(resv["start_utc"]),
                        "now": utc_iso(now),
                    },
                )
            if resv["revision"] != body.expected_revision:
                raise RevisionConflictError(
                    f"expected revision {body.expected_revision} but reservation "
                    f"{reservation_id} is at revision {resv['revision']}",
                    details={
                        "reservation_id": reservation_id,
                        "expected_revision": body.expected_revision,
                        "current_revision": resv["revision"],
                    },
                )

            # Capacity-touching write: expire due locks first and promote the
            # waitlist into the freed slots, in application order.
            _sweep_expired_locks(cur, point, now)

            validate_reservation_window(
                start,
                end,
                now,
                max_days_ahead=settings.reservation_max_days_ahead,
                max_duration=timedelta(
                    minutes=settings.reservation_max_duration_minutes
                ),
            )

            # The new window must fit inside one continuous open interval of
            # the currently published calendar.
            _, rules_row = db.get_current_rules(cur, point_id)
            zone = get_zone(point["timezone"])
            ruleset = payload_to_ruleset(rules_row["payload"])
            date_from = (start - timedelta(days=1)).astimezone(zone).date()
            date_to = (end + timedelta(days=1)).astimezone(zone).date()
            intervals = build_open_intervals(
                point["timezone"], ruleset, date_from, date_to
            )
            if not any(s <= start and end <= e for s, e in intervals):
                raise OutsideOpenHoursError(
                    "rescheduled window is not fully contained in one "
                    "continuous open interval of the published calendar",
                    details={
                        "reservation_id": reservation_id,
                        "start": utc_iso(start),
                        "end": utc_iso(end),
                        "rules_version": rules_row["version"],
                    },
                )

            # Capacity against other active reservations and valid locks;
            # this reservation's own (still held) old window is ignored.
            _assert_window_has_capacity(
                cur, point, start, end, now=now, exclude_reservation_id=resv["id"]
            )

            attempt = db.apply_reschedule(
                cur,
                resv,
                body.idempotency_key,
                body.expected_revision,
                start,
                end,
                rules_row["version"],
            )
            # Moving away frees the old window: in the same transaction,
            # promote every waitlist entry that now fits, in application
            # order.  A failure here rolls the reschedule back as well,
            # so the old window stays occupied.
            db.promote_waitlist(cur, point, rules_row, now)
            # Rescheduling only moves the window and bumps the revision;
            # status / cancel token / created_at are unchanged, so the locked
            # ``resv`` row plus the stored attempt fully describe the result.
            return _reschedule_replay_out(resv, attempt)


def _reassign_out(
    source_row: dict, target_row: dict, target_rules_version: int
) -> ReassignOut:
    """Fresh reassign result: cancelled source + newly booked target."""
    return ReassignOut(
        source_reservation=_reservation_state_out(source_row),
        target_reservation=_reservation_out(target_row),
        target_rules_version=target_rules_version,
    )


def _reassign_replay_out(cur, source_row: dict, attempt: dict) -> ReassignOut:
    """Replay of a successful reassign attempt.

    The target reservation is reported as its *first generated* snapshot
    (original window, target rules version, revision 1 — it is a fresh
    creation at the target point) while its status fields reflect the live
    row, mirroring creation-key replay semantics.  The source reservation is
    reported as it stands now (cancelled by the reassignment).
    """
    target_row = db.get_reservation_by_id(cur, attempt["target_reservation_id"])
    return ReassignOut(
        source_reservation=_reservation_state_out(source_row),
        target_reservation=_creation_replay_out(target_row),
        target_rules_version=attempt["target_rules_version"],
    )


@app.post(
    "/points/{point_id}/reservations/{reservation_id}/reassign",
    response_model=ReassignOut,
    tags=["reservations"],
)
def reassign_reservation(
    point_id: int, reservation_id: int, body: ReservationReassignIn
) -> ReassignOut:
    if body.target_point_id == point_id:
        raise InvalidRequestError(
            "target_point_id must name a different recycling point than the "
            "source; use the reschedule endpoint for same-point moves",
            details={"point_id": point_id, "target_point_id": body.target_point_id},
        )
    start = body.start.astimezone(timezone.utc)
    end = body.end.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Lock both point rows in ascending id order.  Every other
            # mutation takes exactly one point lock, and a reassign in the
            # opposite direction takes the same two locks in the same order,
            # so concurrent reassigns / bookings / publishes serialize
            # without deadlock and cannot double-book a window.
            points = {}
            for pid in sorted({point_id, body.target_point_id}):
                points[pid] = db.lock_point(cur, pid)
            source_point = points[point_id]
            target_point = points[body.target_point_id]

            # 404 / 403 semantics mirror the cancel and reschedule endpoints:
            # existence first, then the capability token.
            resv = db.get_reservation_for_update(cur, point_id, reservation_id)
            db.check_cancel_token(resv, body.cancel_token)

            # Reassign idempotency is scoped to the source reservation: same
            # key + same content replays the first generated target
            # reservation; same key + different content conflicts.  Replay
            # wins over every other state/revision check.
            attempt = db.get_reassign_by_key(
                cur, reservation_id, body.idempotency_key
            )
            if attempt is not None:
                if (
                    attempt["target_point_id"] == body.target_point_id
                    and attempt["start_utc"] == start
                    and attempt["end_utc"] == end
                ):
                    return _reassign_replay_out(cur, resv, attempt)
                raise IdempotencyConflictError(
                    f"reassign idempotency key {body.idempotency_key!r} was "
                    f"already used on reservation {reservation_id} with "
                    f"different content",
                    details={
                        "reservation_id": reservation_id,
                        "target_point_id": attempt["target_point_id"],
                        "start": utc_iso(attempt["start_utc"]),
                        "end": utc_iso(attempt["end_utc"]),
                    },
                )

            if resv["status"] == "cancelled":
                raise ReservationNotReschedulableError(
                    f"reservation {reservation_id} is cancelled and cannot be "
                    f"reassigned",
                    details={
                        "reservation_id": reservation_id,
                        "status": "cancelled",
                        "reason": "reservation_cancelled",
                    },
                )
            if resv["checked_in_at"] is not None:
                # A checked-in visit is being (or has been) consumed: it can
                # no longer be moved to another point, even before the start.
                raise ReservationNotReschedulableError(
                    f"reservation {reservation_id} has checked in and cannot "
                    f"be reassigned",
                    details={
                        "reservation_id": reservation_id,
                        "status": resv["status"],
                        "reason": "reservation_checked_in",
                        "checked_in_at": utc_iso(resv["checked_in_at"]),
                    },
                )
            if resv["start_utc"] <= now:
                raise ReservationNotReschedulableError(
                    f"reservation {reservation_id} has already started and "
                    f"cannot be reassigned",
                    details={
                        "reservation_id": reservation_id,
                        "status": resv["status"],
                        "reason": "reservation_started",
                        "start": utc_iso(resv["start_utc"]),
                        "now": utc_iso(now),
                    },
                )
            if resv["revision"] != body.expected_revision:
                raise RevisionConflictError(
                    f"expected revision {body.expected_revision} but reservation "
                    f"{reservation_id} is at revision {resv['revision']}",
                    details={
                        "reservation_id": reservation_id,
                        "expected_revision": body.expected_revision,
                        "current_revision": resv["revision"],
                    },
                )

            # Capacity-touching on both points: expire due locks first and
            # promote each point's waitlist into the freed slots.
            _sweep_expired_locks(cur, source_point, now)
            _sweep_expired_locks(cur, target_point, now)

            # The target window follows the same booking policy as an
            # ordinary reservation: lead time, duration, one continuous open
            # interval of the target point's currently published calendar, and spare
            # capacity on that point for every instant of the new window.
            validate_reservation_window(
                start,
                end,
                now,
                max_days_ahead=settings.reservation_max_days_ahead,
                max_duration=timedelta(
                    minutes=settings.reservation_max_duration_minutes
                ),
            )
            _, target_rules = db.get_current_rules(cur, body.target_point_id)
            zone = get_zone(target_point["timezone"])
            ruleset = payload_to_ruleset(target_rules["payload"])
            date_from = (start - timedelta(days=1)).astimezone(zone).date()
            date_to = (end + timedelta(days=1)).astimezone(zone).date()
            intervals = build_open_intervals(
                target_point["timezone"], ruleset, date_from, date_to
            )
            if not any(s <= start and end <= e for s, e in intervals):
                raise OutsideOpenHoursError(
                    "reassigned window is not fully contained in one "
                    "continuous open interval of the target point's "
                    "published calendar",
                    details={
                        "reservation_id": reservation_id,
                        "target_point_id": body.target_point_id,
                        "start": utc_iso(start),
                        "end": utc_iso(end),
                        "rules_version": target_rules["version"],
                    },
                )
            _assert_window_has_capacity(cur, target_point, start, end, now=now)

            # All validations passed: cancel the source reservation (this
            # releases its window) and, per the existing release rules, try
            # to promote the origin point's waitlist in the same transaction.
            cancelled, released = db.cancel_reservation(
                cur, point_id, reservation_id, body.cancel_token
            )
            assert released  # status was verified "active" above
            _, source_rules = db.get_current_rules(cur, point_id)
            db.promote_waitlist(cur, source_point, source_rules, now)

            # Book the new reservation at the target point with a
            # server-generated creation key and a fresh cancel token.
            new_reservation = db.insert_reservation(
                cur,
                body.target_point_id,
                f"__reassign__:{reservation_id}:{secrets.token_urlsafe(16)}",
                start,
                end,
                target_rules["version"],
            )
            db.insert_reassign(
                cur,
                reservation_id=reservation_id,
                source_point_id=point_id,
                target_point_id=body.target_point_id,
                idempotency_key=body.idempotency_key,
                expected_revision=body.expected_revision,
                start_utc=start,
                end_utc=end,
                target_reservation_id=new_reservation["id"],
                target_rules_version=target_rules["version"],
            )
            # Any failure above (validation or waitlist promotion) aborts the
            # transaction, leaving the source reservation and the target
            # point's occupancy untouched.
            return _reassign_out(cancelled, new_reservation, target_rules["version"])


# ---------------------------------------------------------------------------
# Waitlist (候补) — applications for windows currently held by another visitor
# ---------------------------------------------------------------------------

def _waitlist_out(cur, entry: dict) -> WaitlistOut:
    """Build the public waitlist view; queue position only while waiting.

    A promoted entry embeds the reservation's *live* row, so its status and
    cancel token reflect later cancellation/rescheduling.
    """
    position = None
    reservation_out = None
    if entry["status"] == "waiting":
        position = db.waitlist_position(cur, entry["point_id"], entry["id"])
    elif entry["reservation_id"] is not None:
        reservation_row = db.get_reservation_by_id(cur, entry["reservation_id"])
        if reservation_row is not None:
            reservation_out = _reservation_out(reservation_row)
    return WaitlistOut(
        id=entry["id"],
        point_id=entry["point_id"],
        status=entry["status"],
        position=position,
        start=utc_iso(entry["start_utc"]),
        end=utc_iso(entry["end_utc"]),
        rules_version=entry["rules_version"],
        query_token=entry["query_token"],
        created_at=entry["created_at"],
        reservation=reservation_out,
    )


@app.post(
    "/points/{point_id}/waitlist",
    response_model=WaitlistOut,
    status_code=201,
    tags=["waitlist"],
)
def create_waitlist_entry(
    point_id: int, body: WaitlistCreateIn, response: Response
) -> WaitlistOut:
    start = body.start.astimezone(timezone.utc)
    end = body.end.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Same per-point lock as creation / cancel / reschedule / rule
            # publishing: application order and every promotion decision are
            # taken against the same serialized state.
            point = db.lock_point(cur, point_id)

            existing = db.get_waitlist_by_key(
                cur, point_id, body.idempotency_key
            )
            if existing is not None:
                if (
                    existing["start_utc"] == start
                    and existing["end_utc"] == end
                ):
                    response.status_code = 200
                    return _waitlist_out(cur, existing)
                raise IdempotencyConflictError(
                    f"waitlist idempotency key {body.idempotency_key!r} was "
                    f"already used on point {point_id} with a different time "
                    f"window",
                    details={
                        "waitlist_id": existing["id"],
                        "start": utc_iso(existing["start_utc"]),
                        "end": utc_iso(existing["end_utc"]),
                    },
                )

            # The application window must itself be a legal booking window
            # (duration / lead time) and must be fully covered by the
            # currently published calendar — same rule version that would
            # govern an ordinary reservation.
            validate_reservation_window(
                start,
                end,
                now,
                max_days_ahead=settings.reservation_max_days_ahead,
                max_duration=timedelta(
                    minutes=settings.reservation_max_duration_minutes
                ),
            )
            # Capacity-touching write: expire due locks first and promote the
            # waitlist into the freed slots, in application order.
            _sweep_expired_locks(cur, point, now)
            _, rules_row = db.get_current_rules(cur, point_id)
            zone = get_zone(point["timezone"])
            ruleset = payload_to_ruleset(rules_row["payload"])
            date_from = (start - timedelta(days=1)).astimezone(zone).date()
            date_to = (end + timedelta(days=1)).astimezone(zone).date()
            intervals = build_open_intervals(
                point["timezone"], ruleset, date_from, date_to
            )
            if not any(s <= start and end <= e for s, e in intervals):
                raise OutsideOpenHoursError(
                    "waitlist window is not fully contained in one continuous "
                    "open interval of the published calendar",
                    details={
                        "start": utc_iso(start),
                        "end": utc_iso(end),
                        "rules_version": rules_row["version"],
                    },
                )

            # The waitlist is only for calendar-valid windows that are
            # already at capacity: some instant of the window must be holding
            # as many occupancy windows (active reservations plus valid
            # locks) as the point allows.  A window with room to spare keeps
            # going through the ordinary reservation flow.  Capacity follows
            # the half-open [start, end) rule, so a window that merely
            # touches active reservations is still free.
            peak = db.window_peak_concurrency(cur, point_id, start, end, now=now)
            if peak < point["capacity"]:
                raise WaitlistWindowAvailableError(
                    "the requested window currently has free capacity; use "
                    "the reservation endpoint instead of the waitlist",
                    details={
                        "start": utc_iso(start),
                        "end": utc_iso(end),
                        "rules_version": rules_row["version"],
                        "capacity": point["capacity"],
                        "peak_concurrency": peak,
                    },
                )

            entry = db.insert_waitlist_entry(
                cur,
                point_id,
                body.idempotency_key,
                start,
                end,
                rules_row["version"],
            )
            return _waitlist_out(cur, entry)


@app.get(
    "/points/{point_id}/waitlist/{entry_id}",
    response_model=WaitlistOut,
    tags=["waitlist"],
)
def read_waitlist_entry(
    point_id: int, entry_id: int, query_token: str | None = None
) -> WaitlistOut:
    if query_token is None:
        # Distinguishable 403 instead of the framework's 422 for a missing
        # capability credential.
        raise InvalidWaitlistTokenError("query_token is required")
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            entry = db.get_waitlist_entry(cur, point_id, entry_id)
            db.check_waitlist_token(entry, query_token)
            return _waitlist_out(cur, entry)


@app.post(
    "/points/{point_id}/waitlist/{entry_id}/withdraw",
    response_model=WaitlistOut,
    tags=["waitlist"],
)
def withdraw_waitlist_entry(
    point_id: int, entry_id: int, body: WaitlistTokenIn
) -> WaitlistOut:
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Point lock first, then the entry row: same lock order as every
            # release sweep, so withdrawal cannot deadlock against a promotion.
            db.lock_point(cur, point_id)
            entry = db.withdraw_waitlist_entry(
                cur, point_id, entry_id, body.query_token
            )
            return _waitlist_out(cur, entry)


# ---------------------------------------------------------------------------
# Slot locks (到访名额锁定) — short-TTL holds confirmed into reservations
# ---------------------------------------------------------------------------

def _lock_out(cur, lock: dict, now: datetime) -> LockOut:
    """Build the public lock view.

    A row still marked ``locked`` whose confirmation deadline has passed
    reads as ``expired`` (the next capacity-touching write will persist that
    state); a confirmed lock embeds the reservation's *live* row, so its
    status and cancel token reflect later cancellation/rescheduling.
    """
    status = lock["status"]
    if status == "locked" and lock["expires_at"] <= now:
        status = "expired"
    reservation_out = None
    if lock["reservation_id"] is not None:
        reservation_row = db.get_reservation_by_id(cur, lock["reservation_id"])
        if reservation_row is not None:
            reservation_out = _reservation_out(reservation_row)
    return LockOut(
        id=lock["id"],
        point_id=lock["point_id"],
        status=status,
        start=utc_iso(lock["start_utc"]),
        end=utc_iso(lock["end_utc"]),
        rules_version=lock["rules_version"],
        confirm_token=lock["confirm_token"],
        expires_at=lock["expires_at"],
        created_at=lock["created_at"],
        reservation=reservation_out,
    )


@app.post(
    "/points/{point_id}/locks",
    response_model=LockOut,
    status_code=201,
    tags=["locks"],
)
def create_lock(point_id: int, body: LockCreateIn, response: Response) -> LockOut:
    start = body.start.astimezone(timezone.utc)
    end = body.end.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Same per-point lock as reservation creation / rule publishing:
            # lock creation serializes with every other capacity decision.
            point = db.lock_point(cur, point_id)

            # Idempotency: the first successful request wins.  A same-content
            # retry replays the stored lock row (its status reflects the
            # current state, including a later confirmation or expiry);
            # different content under the same key conflicts.
            existing = db.get_lock_by_key(cur, point_id, body.idempotency_key)
            if existing is not None:
                if existing["start_utc"] == start and existing["end_utc"] == end:
                    response.status_code = 200
                    return _lock_out(cur, existing, now)
                raise IdempotencyConflictError(
                    f"lock idempotency key {body.idempotency_key!r} was "
                    f"already used on point {point_id} with a different time "
                    f"window",
                    details={
                        "lock_id": existing["id"],
                        "start": utc_iso(existing["start_utc"]),
                        "end": utc_iso(existing["end_utc"]),
                    },
                )

            # Capacity-touching write: expire due locks first and promote the
            # waitlist into the freed slots, in application order.
            _sweep_expired_locks(cur, point, now)

            # The lock window follows the single-reservation booking policy:
            # lead time / duration, one continuous open interval of the
            # currently published calendar, and spare capacity for every
            # instant of [start, end) — where valid locks count as occupancy
            # alongside active reservations.
            validate_reservation_window(
                start,
                end,
                now,
                max_days_ahead=settings.reservation_max_days_ahead,
                max_duration=timedelta(
                    minutes=settings.reservation_max_duration_minutes
                ),
            )
            _, rules_row = db.get_current_rules(cur, point_id)
            zone = get_zone(point["timezone"])
            ruleset = payload_to_ruleset(rules_row["payload"])
            date_from = (start - timedelta(days=1)).astimezone(zone).date()
            date_to = (end + timedelta(days=1)).astimezone(zone).date()
            intervals = build_open_intervals(
                point["timezone"], ruleset, date_from, date_to
            )
            if not any(s <= start and end <= e for s, e in intervals):
                raise OutsideOpenHoursError(
                    "lock window is not fully contained in one continuous "
                    "open interval of the published calendar",
                    details={
                        "start": utc_iso(start),
                        "end": utc_iso(end),
                        "rules_version": rules_row["version"],
                    },
                )
            _assert_window_has_capacity(cur, point, start, end, now=now)

            lock = db.insert_lock(
                cur,
                point_id,
                body.idempotency_key,
                start,
                end,
                rules_row["version"],
                now + timedelta(minutes=settings.lock_ttl_minutes),
            )
            return _lock_out(cur, lock, now)


@app.post(
    "/points/{point_id}/locks/{lock_id}/confirm",
    response_model=LockOut,
    tags=["locks"],
)
def confirm_lock(point_id: int, lock_id: int, body: LockConfirmIn) -> LockOut:
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Same per-point lock as every other mutation: the confirm and a
            # concurrent waitlist promotion (triggered by another write)
            # serialize, so the expiry instant can never double-occupy.
            point = db.lock_point(cur, point_id)

            # 404 / 403 semantics mirror the cancel endpoint: existence
            # first, then the capability token.
            lock = db.get_lock_for_update(cur, point_id, lock_id)
            db.check_confirm_token(lock, body.confirm_token)

            # Confirmation is naturally idempotent: a repeat confirm returns
            # the same reservation and never extends the lock's deadline.
            if lock["status"] == "confirmed":
                return _lock_out(cur, lock, now)
            if lock["status"] == "expired" or lock["expires_at"] <= now:
                raise LockExpiredError(
                    f"lock {lock_id} expired at {utc_iso(lock['expires_at'])} "
                    f"without being confirmed and can no longer convert",
                    details={
                        "lock_id": lock_id,
                        "expires_at": utc_iso(lock["expires_at"]),
                        "now": utc_iso(now),
                    },
                )

            # Capacity-touching write: expire other due locks first and
            # promote the waitlist into the freed slots, in application
            # order.  This lock is still valid (checked above), so the sweep
            # cannot touch it, and the promotion cannot take its slot — the
            # lock itself counts as occupancy during the sweep.
            _sweep_expired_locks(cur, point, now)

            # Atomically convert the hold into an ordinary reservation: the
            # slot the lock was holding simply changes kind (one lock slot
            # becomes one reservation slot), so no capacity check is needed.
            # The reservation is validated against the rules version current
            # at confirmation time — calendar changes that would un-cover the
            # lock were refused while it was valid, so its window is still
            # open under that version.
            _, rules_row = db.get_current_rules(cur, point_id)
            reservation = db.insert_reservation(
                cur,
                point_id,
                f"__lock_confirm__:{lock_id}:{secrets.token_urlsafe(16)}",
                lock["start_utc"],
                lock["end_utc"],
                rules_row["version"],
            )
            lock = db.mark_lock_confirmed(cur, lock_id, reservation["id"])
            return _lock_out(cur, lock, now)


# ---------------------------------------------------------------------------
# Trips (行程) — atomic multi-stop bookings across recycling points
# ---------------------------------------------------------------------------

def _trip_out(trip: dict, stops: list[TripStopOut]) -> TripOut:
    return TripOut(
        trip_id=trip["id"],
        idempotency_key=trip["idempotency_key"],
        created_at=trip["created_at"],
        stops=stops,
    )


def _trip_replay_out(cur, trip: dict, stored_stops: list[dict]) -> TripOut:
    """First-creation snapshot of a booked trip.

    Every stop reports its reservation as first created (original window,
    rules version and revision 1) while the status fields reflect the live
    reservation row — mirroring creation-key replay semantics, so a replay
    after a stop was rescheduled or cancelled still returns the original
    group content.
    """
    stops = []
    for stored in stored_stops:
        reservation = db.get_reservation_by_id(cur, stored["reservation_id"])
        stops.append(
            TripStopOut(
                point_id=stored["point_id"],
                reservation=_creation_replay_out(reservation),
            )
        )
    return _trip_out(trip, stops)


def _trip_content_matches(stored_stops: list[dict], body: TripCreateIn) -> bool:
    """A retry matches only the complete ordered stop content of the trip."""
    if len(stored_stops) != len(body.stops):
        return False
    for stored, stop in zip(stored_stops, body.stops):
        if (
            stored["point_id"] != stop.point_id
            or stored["start_utc"] != stop.start.astimezone(timezone.utc)
            or stored["end_utc"] != stop.end.astimezone(timezone.utc)
        ):
            return False
    return True


def _stop_scoped(error: ApiError, stop_index: int, point_id: int) -> ApiError:
    """Re-raise a per-stop failure tagged with the stop that caused it.

    The original error type (HTTP status, code) and message are preserved;
    ``stop_index`` (0-based position in the submitted ``stops`` array) and
    ``point_id`` are added to the details so the caller can tell which stop
    of the group failed and why nothing was booked.
    """
    details = {"stop_index": stop_index, "point_id": point_id, **error.details}
    return type(error)(
        f"trip stop {stop_index} (point {point_id}): {error.message}",
        details=details,
    )


@app.post(
    "/trips",
    response_model=TripOut,
    status_code=201,
    tags=["trips"],
)
def create_trip(body: TripCreateIn, response: Response) -> TripOut:
    # Structural rules of one itinerary (checked before any idempotency
    # lookup, mirroring the reassign endpoint's same-point check): every
    # stop visits a different recycling point, and the stops are in
    # chronological visit order — a stop must end no later than the next
    # one begins (touching is allowed).
    index_by_point: dict[int, int] = {}
    for index, stop in enumerate(body.stops):
        first = index_by_point.get(stop.point_id)
        if first is not None:
            raise InvalidRequestError(
                f"recycling point {stop.point_id} appears more than once in "
                f"one trip (stops {first} and {index}); each stop of a trip "
                f"must visit a different point",
                details={
                    "point_id": stop.point_id,
                    "stop_indexes": [first, index],
                },
            )
        index_by_point[stop.point_id] = index
    windows = [
        (stop.start.astimezone(timezone.utc), stop.end.astimezone(timezone.utc))
        for stop in body.stops
    ]
    for index in range(1, len(windows)):
        previous_end = windows[index - 1][1]
        start = windows[index][0]
        if previous_end > start:
            raise InvalidPeriodError(
                f"trip stop {index} starts at {utc_iso(start)} before stop "
                f"{index - 1} ends at {utc_iso(previous_end)}; stops must be "
                f"in visit order and each stop must end no later than the "
                f"next one begins",
                details={
                    "stop_index": index,
                    "previous_stop_index": index - 1,
                    "previous_end": utc_iso(previous_end),
                    "start": utc_iso(start),
                },
            )
    now = datetime.now(timezone.utc)
    with db.conn() as connection:
        with connection.transaction():
            cur = connection.cursor()
            # Claim the trip key first: a concurrent request with the same
            # key blocks here until this transaction resolves, then replays
            # or conflicts against the committed content.  A trip that fails
            # validation rolls this insert back, so a failed attempt never
            # consumes its key.
            trip = db.insert_trip_key(cur, body.idempotency_key)
            if trip is None:
                existing = db.get_trip_by_key(cur, body.idempotency_key)
                stored_stops = db.get_trip_stops(cur, existing["id"])
                if _trip_content_matches(stored_stops, body):
                    response.status_code = 200
                    return _trip_replay_out(cur, existing, stored_stops)
                raise IdempotencyConflictError(
                    f"trip idempotency key {body.idempotency_key!r} was "
                    f"already used with different stop content",
                    details={
                        "trip_id": existing["id"],
                        "stops": [
                            {
                                "point_id": stored["point_id"],
                                "start": utc_iso(stored["start_utc"]),
                                "end": utc_iso(stored["end_utc"]),
                            }
                            for stored in stored_stops
                        ],
                    },
                )
            trip_id = trip["id"]

            # Lock every involved point row in ascending id order — the same
            # order reassigns use, and a superset of the single lock that
            # creations, cancels, reschedules, publishes and waitlist
            # promotions take — so concurrent trips (even with reversed stop
            # order), bookings and rule changes serialize without deadlock
            # and cannot double-book a window.
            points = {}
            for point_id in sorted(index_by_point):
                try:
                    points[point_id] = db.lock_point(cur, point_id)
                except NotFoundError as exc:
                    raise _stop_scoped(exc, index_by_point[point_id], point_id)

            # Capacity-touching on every involved point: expire due locks
            # first and promote each point's waitlist into the freed slots.
            for point in points.values():
                _sweep_expired_locks(cur, point, now)

            # Validate every stop in visit order against the same rules as a
            # single reservation (booking window, one continuous open
            # interval of the currently published calendar, per-point
            # capacity over [start, end)).  The first failure aborts the
            # whole group: the transaction rolls back and nothing is stored.
            plan = []
            for index, stop in enumerate(body.stops):
                start, end = windows[index]
                point = points[stop.point_id]
                try:
                    validate_reservation_window(
                        start,
                        end,
                        now,
                        max_days_ahead=settings.reservation_max_days_ahead,
                        max_duration=timedelta(
                            minutes=settings.reservation_max_duration_minutes
                        ),
                    )
                    _, rules_row = db.get_current_rules(cur, stop.point_id)
                    zone = get_zone(point["timezone"])
                    ruleset = payload_to_ruleset(rules_row["payload"])
                    date_from = (start - timedelta(days=1)).astimezone(zone).date()
                    date_to = (end + timedelta(days=1)).astimezone(zone).date()
                    intervals = build_open_intervals(
                        point["timezone"], ruleset, date_from, date_to
                    )
                    if not any(s <= start and end <= e for s, e in intervals):
                        raise OutsideOpenHoursError(
                            "reservation is not fully contained in one "
                            "continuous open interval of the published "
                            "calendar",
                            details={
                                "start": utc_iso(start),
                                "end": utc_iso(end),
                                "rules_version": rules_row["version"],
                            },
                        )
                    _assert_window_has_capacity(cur, point, start, end, now=now)
                except ApiError as exc:
                    raise _stop_scoped(exc, index, stop.point_id)
                plan.append(
                    (index, stop.point_id, start, end, rules_row["version"])
                )

            # Every stop validated: book them all in this one transaction,
            # each with a server-generated per-point reservation key.
            stops_out = []
            for index, point_id, start, end, rules_version in plan:
                reservation = db.insert_reservation(
                    cur,
                    point_id,
                    f"__trip__:{trip_id}:{index}:{secrets.token_urlsafe(16)}",
                    start,
                    end,
                    rules_version,
                )
                db.insert_trip_stop(
                    cur,
                    trip_id=trip_id,
                    position=index,
                    point_id=point_id,
                    reservation_id=reservation["id"],
                    start_utc=start,
                    end_utc=end,
                    rules_version=rules_version,
                )
                stops_out.append(
                    TripStopOut(
                        point_id=point_id,
                        reservation=_reservation_out(reservation),
                    )
                )
            return _trip_out(trip, stops_out)


# ---------------------------------------------------------------------------
# Trip planning (行程规划搜索) — read-only earliest itinerary search
# ---------------------------------------------------------------------------


@app.post(
    "/trips/plan",
    response_model=TripPlanOut,
    tags=["trips"],
)
def plan_trip(body: TripPlanIn) -> TripPlanOut:
    """Locate the earliest feasible 2..5-stop itinerary without booking it.

    Pure read: no reservation, trip row or placeholder is created, nothing is
    locked, and the answer does not guarantee that a later booking still has
    capacity.  One query answers from one REPEATABLE READ database snapshot.
    """
    # Structural itinerary rules, checked in the same order as trip booking:
    # distinct points, then a well-formed search window, then the per-stop
    # stay-length policy.
    index_by_point: dict[int, int] = {}
    for index, stop in enumerate(body.stops):
        first = index_by_point.get(stop.point_id)
        if first is not None:
            raise InvalidRequestError(
                f"recycling point {stop.point_id} appears more than once in "
                f"one trip plan (stops {first} and {index}); each stop must "
                f"name a different point",
                details={
                    "point_id": stop.point_id,
                    "stop_indexes": [first, index],
                },
            )
        index_by_point[stop.point_id] = index

    search_from = body.search_from.astimezone(timezone.utc)
    search_to = body.search_to.astimezone(timezone.utc)
    if search_to < search_from:
        raise InvalidPeriodError(
            "search_to must be on or after search_from",
            details={
                "search_from": utc_iso(search_from),
                "search_to": utc_iso(search_to),
            },
        )
    span = search_to - search_from
    max_span = timedelta(days=settings.trip_search_max_days)
    if span > max_span:
        raise InvalidPeriodError(
            f"the search window spans {span}; the maximum allowed trip-search "
            f"span is {settings.trip_search_max_days} days",
            details={
                "search_from": utc_iso(search_from),
                "search_to": utc_iso(search_to),
                "span_days": span.total_seconds() / 86400,
                "max_trip_search_days": settings.trip_search_max_days,
            },
        )

    max_duration = timedelta(
        minutes=settings.reservation_max_duration_minutes
    )
    for index, stop in enumerate(body.stops):
        duration = timedelta(minutes=stop.duration_minutes)
        if duration > max_duration:
            raise InvalidPeriodError(
                f"trip-plan stop {index} lasts {stop.duration_minutes} "
                f"minute(s); the existing per-stop reservation limit is "
                f"{settings.reservation_max_duration_minutes} minutes",
                details={
                    "stop_index": index,
                    "point_id": stop.point_id,
                    "duration_minutes": stop.duration_minutes,
                    "max_duration_minutes":
                        settings.reservation_max_duration_minutes,
                },
            )

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=settings.reservation_max_days_ahead)
    # The whole group must stay inside both the search window and the existing
    # bookable horizon; reservation starts must also not be in the past.
    lower = max(search_from, now)
    upper_end = min(search_to, horizon)

    # One repeatable-read snapshot for every point, rule version and active
    # reservation the answer is derived from.  This is a read-only query and
    # takes no point locks; a concurrent booking may still change capacity
    # afterwards (the response is advisory and books nothing).
    specs: list[StopSpec] = []
    with db.conn() as connection:
        previous_level = connection.isolation_level
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        try:
            with connection.transaction():
                cur = connection.cursor()
                for index, stop in enumerate(body.stops):
                    try:
                        point, rules_row = db.get_current_rules(
                            cur, stop.point_id
                        )
                    except ApiError as exc:
                        # Missing point (404) and unpublished rules (409)
                        # keep their distinguishable codes; the failing stop
                        # is named in the details.
                        raise _stop_scoped(exc, index, stop.point_id)
                    windows = (
                        db.get_active_reservation_windows(
                            cur, stop.point_id, lower, upper_end, now
                        )
                        if lower < upper_end
                        else []
                    )
                    specs.append(
                        StopSpec(
                            point_id=stop.point_id,
                            timezone=point["timezone"],
                            capacity=point["capacity"],
                            rules_version=rules_row["version"],
                            ruleset=payload_to_ruleset(rules_row["payload"]),
                            reservations=tuple(windows),
                        )
                    )
        finally:
            connection.isolation_level = previous_level

    durations = [
        timedelta(minutes=stop.duration_minutes) for stop in body.stops
    ]
    gap = timedelta(minutes=body.min_gap_minutes)
    planned = plan_itinerary(
        specs,
        durations,
        gap,
        search_from=search_from,
        search_to=search_to,
        now=now,
        horizon=horizon,
    )
    return TripPlanOut(
        search_from=utc_iso(search_from),
        search_to=utc_iso(search_to),
        min_gap_minutes=body.min_gap_minutes,
        stops=[
            TripPlanStopOut(
                point_id=item.point_id,
                start=utc_iso(item.start),
                end=utc_iso(item.end),
                duration_minutes=int((item.end - item.start).total_seconds() // 60),
                rules_version=item.rules_version,
            )
            for item in (planned or [])
        ],
    )
