import asyncio
import functools


def async_to_sync(func):
    """
    A decorator that converts an asynchronous function to a synchronous function.

    This decorator can be used to call an asynchronous function from a synchronous context.
    If the function is already running in an async environment, it will be called directly.
    Otherwise, it will create a new event loop and run the function using `asyncio.run()`.

    Args:
        func: The asynchronous function to be converted.

    Returns:
        The synchronous wrapper function.

    Raises:
        RuntimeError: If no event loop is running and the function is not called from an async context.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No running loop
            loop = None

        if loop and loop.is_running():
            # asyncio.run() cannot be called from a running event loop
            # if we're here, means we're already in an async environment
            return func(*args, **kwargs)
        else:
            # Either no event loop is running, or we're not inside async context
            return asyncio.run(func(*args, **kwargs))

    return wrapper
