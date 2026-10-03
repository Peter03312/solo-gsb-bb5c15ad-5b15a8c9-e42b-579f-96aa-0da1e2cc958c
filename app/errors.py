"""Distinguishable API errors.

Each error carries a stable machine-readable ``code`` so clients can tell the
mandatory failure cases apart (invalid period / unknown timezone / no
published rules / version conflict / not found).
"""

from __future__ import annotations


class ApiError(Exception):
    status_code: int = 400
    code: str = "bad_request"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(ApiError):
    status_code = 404
    code = "point_not_found"


class VersionNotFoundError(ApiError):
    status_code = 404
    code = "version_not_found"


class VersionConflictError(ApiError):
    status_code = 409
    code = "version_conflict"


class NoPublishedVersionError(ApiError):
    status_code = 409
    code = "no_published_version"


class InvalidPeriodError(ApiError):
    status_code = 422
    code = "invalid_period"


class UnknownTimezoneError(ApiError):
    status_code = 422
    code = "unknown_timezone"


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------

class ReservationNotFoundError(ApiError):
    status_code = 404
    code = "reservation_not_found"


class InvalidCancelTokenError(ApiError):
    status_code = 403
    code = "invalid_cancel_token"


class IdempotencyConflictError(ApiError):
    """Same point + idempotency key but a different request payload."""

    status_code = 409
    code = "idempotency_conflict"


class ReservationOverlapError(ApiError):
    """The requested window overlaps another active reservation."""

    status_code = 409
    code = "reservation_overlap"


class OutsideOpenHoursError(ApiError):
    """The requested window is not fully inside one continuous open interval."""

    status_code = 409
    code = "outside_open_hours"


class ReservationChangeConflictError(ApiError):
    """A rule publish / timezone change would un-cover active reservations."""

    status_code = 409
    code = "reservation_change_conflict"


class CapacityConflictError(ApiError):
    """A capacity decrease below future non-cancelled concurrency is refused."""

    status_code = 409
    code = "capacity_conflict"


class RevisionConflictError(ApiError):
    """Reschedule expected_revision does not match the reservation's revision."""

    status_code = 409
    code = "revision_conflict"


class ReservationNotReschedulableError(ApiError):
    """The reservation cannot be rescheduled (cancelled or already started)."""

    status_code = 409
    code = "reservation_not_reschedulable"


class CheckinNotAllowedError(ApiError):
    """Check-in (签到) is not allowed for this reservation right now."""

    status_code = 409
    code = "checkin_not_allowed"


class CheckoutNotAllowedError(ApiError):
    """Check-out (离场) is not allowed for this reservation right now."""

    status_code = 409
    code = "checkout_not_allowed"


class LockNotConfirmedError(ApiError):
    """The number names a slot lock that has not been confirmed yet."""

    status_code = 409
    code = "lock_not_confirmed"


class InvalidRequestError(ApiError):
    """The request is structurally valid JSON but semantically unacceptable."""

    status_code = 422
    code = "invalid_request"


# ---------------------------------------------------------------------------
# Waitlist (候补)
# ---------------------------------------------------------------------------


class WaitlistEntryNotFoundError(ApiError):
    """Waitlist number does not exist (or does not belong to this point)."""

    status_code = 404
    code = "waitlist_entry_not_found"


class InvalidWaitlistTokenError(ApiError):
    """Status query / withdrawal presented no or a wrong capability token."""

    status_code = 403
    code = "invalid_waitlist_token"


class WaitlistWindowAvailableError(ApiError):
    """The requested window is currently bookable — waitlist does not apply."""

    status_code = 409
    code = "waitlist_window_available"


class WaitlistEntryNotActiveError(ApiError):
    """Withdrawal target is no longer waiting (promoted/expired/withdrawn)."""

    status_code = 409
    code = "waitlist_entry_not_active"


# ---------------------------------------------------------------------------
# Slot locks (到访名额锁定)
# ---------------------------------------------------------------------------


class LockNotFoundError(ApiError):
    """Lock number does not exist (or does not belong to this point)."""

    status_code = 404
    code = "lock_not_found"


class InvalidConfirmTokenError(ApiError):
    """Confirmation presented no or a wrong capability token."""

    status_code = 403
    code = "invalid_confirm_token"


class LockExpiredError(ApiError):
    """The lock passed its confirmation deadline and can no longer convert."""

    status_code = 409
    code = "lock_expired"
