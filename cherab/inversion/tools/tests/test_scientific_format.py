import pytest

from cherab.inversion.tools.scientific_format import parse_scientific_notation


def test_parse_scientific_notation():
    # Test case 1: Valid scientific notation string
    assert parse_scientific_notation("1.23e-4") == "1.23 \\times 10^{-4}"

    # Test case 2: Scientific notation string with positive exponent
    assert parse_scientific_notation("2.5e3") == "2.5 \\times 10^{3}"

    # Test case 3: Scientific notation string with custom scilimits
    assert parse_scientific_notation("3.14e-2", scilimits=(-2, 2)) == "0.0314"
    assert parse_scientific_notation("3.14e+01", scilimits=(-1, 1)) == "31.40"

    # Test case 4: Scientific notation string with LaTeX formatting
    assert parse_scientific_notation("5.67e-8", useLatex=True) == "5.67 \\times 10^{-8}"

    # Test case 5: Scientific notation string with disabled LaTeX formatting
    assert parse_scientific_notation("6.78e-9", useLatex=False) == "6.78x10^-9"

    # Test case 6: Scientific notation string with 00 exponent
    assert parse_scientific_notation("9.0e00") == "9.0"
    assert parse_scientific_notation("9.0e00", scilimits=(-1, -1)) == "9.0 \\times 10^{0}"

    # Test case 7: Scientific notation string with invalid string argument
    with pytest.raises(ValueError):
        parse_scientific_notation("1.23-4.5")
