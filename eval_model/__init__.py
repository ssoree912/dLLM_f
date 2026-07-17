from .LLaDA import LLaDA

try:
    from .Dream import Dream
except ModuleNotFoundError as exc:
    _dream_import_error = exc

    class Dream:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Dream requires an optional dependency that is not installed"
            ) from _dream_import_error

__all__ = ["Dream", "LLaDA"]
