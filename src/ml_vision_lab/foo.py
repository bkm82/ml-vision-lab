def foo(bar: str) -> str:
    """Summary line.

    Extended description of function.

    Args:
        bar: Description of input argument.

    Returns:
        Description of return value
    """

    return bar


def my_add(a: int, b: int, absolute: bool = False) -> int:
    """Add two integers together.

    A simple function to test the structure of the overall project

    Args:
        a:int  the first integer.
        b:int  the second integer.
        absolute: bool if the absolute value should be returned

    Returns:
        Returns the sum of the two integers.
        If absolute = True then return the aboslute value of the sum.
    """
    result = a + b
    if absolute:
        result = abs(result)
    return result


if __name__ == "__main__":  # pragma: no cover
    pass
