def foo(bar: str) -> str:
    """Summary line.

    Extended description of function.

    Args:
        bar: Description of input argument.

    Returns:
        Description of return value
    """

    return bar


def my_add(a: int, b: int) -> int:
    """Add two integers together.

    A simple function to test the structure of the overall project

    Args:
        a:int  the first integer.
        b:int  the second integer.

    Returns:
        Returns the absolute value of the sum of the two integers.
    """

    return abs(a + b)


if __name__ == "__main__":  # pragma: no cover
    pass
