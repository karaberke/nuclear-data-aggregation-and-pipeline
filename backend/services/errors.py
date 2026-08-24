"""Transport-independent domain errors.

`charts.py` used to raise `fastapi.HTTPException` directly. That is fine for a
route, but a Dash callback catching a Starlette exception to build a user
message would couple the UI to the HTTP framework. Services now raise these
instead; the routes translate them back into `HTTPException` at the boundary,
and Dash callbacks render `str(error)` inline.
"""


class DomainError(Exception):
    """An expected, user-reportable failure. The message is safe to display."""

    status_code = 500


class SelectionUnavailableError(DomainError):
    """A requested database/isotope/dataset is not offered by JANIS."""

    status_code = 422


class DataUnavailableError(DomainError):
    """JANIS returned no usable rows for an otherwise valid selection."""

    status_code = 404


class JanisError(DomainError):
    """JANIS failed, timed out, or was refused admission by the queue gate."""

    status_code = 500
