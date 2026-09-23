"""Helpers for keeping user-controlled paths inside an intended directory."""

from pathlib import PurePosixPath, PureWindowsPath
import os
import unicodedata

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", filename)
    if normalized in {".", ".."}:
        raise ValueError("dot path components are not valid filenames")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError("control characters are not valid in filenames")
    if any(separator in normalized for separator in ("/", "\\", "\x00", ":")):
        raise ValueError("path separators are not valid in filenames")
    if PurePosixPath(normalized).is_absolute():
        raise ValueError("absolute paths are not valid filenames")
    windows_path = PureWindowsPath(normalized)
    if windows_path.is_absolute() or windows_path.drive:
        raise ValueError("drive and UNC paths are not valid filenames")
    if normalized.rstrip(" .") != normalized:
        raise ValueError("trailing dots and spaces are not valid in filenames")
    if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("reserved device names are not valid filenames")
    return filename


def safe_join(base_dir: str, *parts: str) -> str:
    base = os.path.realpath(os.path.abspath(base_dir))
    candidate = os.path.realpath(os.path.join(base, *parts))
    if os.path.commonpath((base, candidate)) != base:
        raise ValueError("path escapes its permitted directory")
    return candidate
