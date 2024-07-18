"""This module provides a function for parsing a string in scientific notation."""

from decimal import Decimal

__all__ = ["parse_scientific_notation"]


def parse_scientific_notation(sci_str: str, scilimits: tuple = (-1, 1), useLatex=True) -> str:
    """Parse a string in scientific notation and return a formatted string.

    Strings like `'1.23e-4'` are converted to `'1.23x10^-4'` or `'1.23 \\\\times 10^{-4}'`
    if `useLatex` is True.

    Parameters
    ----------
    sci_str : str
        A string in scientific notation.
        Must be in the form of 'mantissa' + 'e' + 'exponent'.
    scilimits : tuple, (m, n), optional
        Scientific notation parsing is conducted only for numbers outside the range 10 :sup:`m` to
        10 :sup:`n`, by default (-1, 1).
    useLatex : bool, optional
        If True, the output string will be formatted using LaTeX.

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    >>> parse_scientific_notation('1.23e-4')
    '1.23 \\\\times 10^{-4}'
    """
    # find the index of the 'e' character
    e_index = sci_str.find("e")

    if e_index == -1:
        raise ValueError(f"There is no 'e' character in the string '{sci_str}'")

    # extract the mantissa, exponent sign and exponent
    mantissa = sci_str[:e_index]
    exponent_sign = sci_str[e_index + 1]
    if exponent_sign not in {"-", "+"}:
        exponent_sign = "+"
        e_index -= 1
    exponent = sci_str[e_index + 2 :]

    # transform the string exponent to an decimal number
    exponent = Decimal(exponent)

    # add a "-" sign to the exponent if it is negative
    if exponent_sign == "-":
        exponent = -exponent

    # check power limits
    lower_limit, upper_limit = scilimits

    if exponent < lower_limit or upper_limit < exponent:
        if useLatex:
            return f"{mantissa} \\times 10^{{{exponent}}}"
        else:
            return f"{mantissa}x10^{exponent}"
    else:
        # if the exponent is within the limits, return the float form
        return f"{Decimal(mantissa) * Decimal('10') ** exponent}"
