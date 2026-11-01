"""Typed errors that are surfaced to the UI with actionable messages."""

from __future__ import annotations


class LumenError(Exception):
    """Base class for user-facing errors.

    ``code`` is a stable machine-readable identifier used by the frontend,
    ``message`` is a human readable explanation and ``hint`` tells the user
    what they can do about it.
    """

    code = "error"
    status = 400

    def __init__(self, message: str, hint: str | None = None, **details):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint, "details": self.details}

    def __reduce__(self):
        # keep message/hint/details when the error crosses a process boundary
        return (_restore_error, (type(self), self.message, self.hint, self.details))


def _restore_error(cls, message, hint, details):
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    err.message, err.hint, err.details = message, hint, details
    return err


class ModelNotInstalledError(LumenError):
    code = "model_not_installed"
    status = 409

    def __init__(self, model_id: str, name: str):
        super().__init__(
            f"The model '{name}' is not installed on this computer.",
            hint="Open Models and install it. Downloading needs an internet connection once; "
            "after that, processing works completely offline. You can also import the model file manually.",
            model_id=model_id,
        )


class InvalidInputError(LumenError):
    code = "invalid_input"


class NoFaceFoundError(LumenError):
    code = "no_face"


class IncompatibleInputsError(LumenError):
    """Raised when inputs cannot produce an honest, realistic result."""

    code = "incompatible"
    status = 422


class ProcessingError(LumenError):
    code = "processing_failed"
    status = 500


class NotFoundError(LumenError):
    code = "not_found"
    status = 404
