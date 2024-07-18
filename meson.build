project(
  'cherab-inversion',
  # 'c',
  # 'cython',
  version: run_command(
    ['cherab/inversion/_build_utils/gitversion.py'],
    check: true
  ).stdout().strip(),
  meson_version: '>= 0.64.0',
  default_options: [
    'c_std=c99',
    'buildtype=debugoptimized',
  ],
)

fs = import('fs')
py = import('python').find_installation(pure: true)

# Python dependency
py_dep = py.dependency()

# OpenMP dependency
# NOTE: uncomment if you want to use prange in cython
# omp_dep = dependency('openmp')

# NumPy dependency
# incdir_numpy = run_command(
#   py,
#   [
#     '-c',
#     'import numpy; print(numpy.get_include())'
#   ],
#   check: true
# ).stdout().strip()
# inc_np = include_directories(incdir_numpy)
# np_dep = declare_dependency(include_directories: inc_np)

# Numpy deprecated api (if errors is occurred, comment out in module section)
# numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# subdirectories
subdir('cherab/inversion')
