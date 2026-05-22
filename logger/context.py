#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Request context management.

Stores per-request metadata (e.g. request_id) using contextvars,
which works correctly in both sync and async FastAPI handlers.
"""

from contextvars import ContextVar, Token
from typing import Any

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_task_results_var: ContextVar[list[dict[str, Any]]] = ContextVar("task_results", default=[])
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def set_request_id(request_id: str) -> Token:
    """Set the request_id for the current context and return the reset token."""
    return _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the request_id for the current context, or '-' if not set."""
    return _request_id_var.get()


def reset_request_id(token: Token) -> None:
    """Reset the request_id to its previous value using the token from set_request_id."""
    _request_id_var.reset(token)


def set_session_id(session_id: str) -> Token:
    """Set the session_id for the current context and return the reset token."""
    return _session_id_var.set(session_id)


def get_session_id() -> str:
    """Return the session_id for the current context, or '' if not set."""
    return _session_id_var.get()


def reset_session_id(token: Token) -> None:
    """Reset the session_id to its previous value using the token from set_session_id."""
    _session_id_var.reset(token)


def set_user_id(user_id: str) -> Token:
    """Set the user_id for the current context and return the reset token."""
    return _user_id_var.set(user_id)


def get_user_id() -> str:
    """Return the user_id for the current context, or '' if not set."""
    return _user_id_var.get()


def reset_user_id(token: Token) -> None:
    """Reset the user_id to its previous value using the token from set_user_id."""
    _user_id_var.reset(token)


def set_task_results(results: list[dict[str, Any]]) -> Token:
    """Set the task_results for the current context and return the reset token."""
    return _task_results_var.set(results)


def get_task_results() -> list[dict[str, Any]]:
    """Return the task_results for the current context, or [] if not set."""
    return _task_results_var.get()


def append_task_results(results: list[dict[str, Any]]) -> None:
    """Append task results to the current context's task_results list."""
    current = _task_results_var.get()
    current.extend(results)


def reset_task_results(token: Token) -> None:
    """Reset the task_results to its previous value."""
    _task_results_var.reset(token)
