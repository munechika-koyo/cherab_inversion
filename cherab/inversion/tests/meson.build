python_sources = [
    '__init__.py',
    'conftest.py',
    'test_core.py',
    'test_gcv.py',
    'test_lcurve.py',
    'test_mfr.py',
    'test_derivative.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/inversion/tests',
)

py.install_sources(
  [
    'data/raytransfer_grid_data.npz',
  ],
  subdir: 'cherab/inversion/tests/data',
)
