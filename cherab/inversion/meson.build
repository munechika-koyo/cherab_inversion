cython_sources = [
  ['derivative', 'derivative.pyx'],
]

# compile cython sources to .so modules
foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: [py_dep, np_dep],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/inversion',
  )
endforeach


py.install_sources(
  [
    '__init__.py',
    'core.py',
    'gcv.py',
    'lcurve.py',
    'mfr.py',
  ],
  subdir: 'cherab/inversion',
)

# Generate version.py for sdist
meson.add_dist_script(
  ['_build_utils/gitversion.py', '--meson-dist', '--write', 'cherab/inversion/version.py']
)
if not fs.exists('cherab/inversion/version.py')
  generate_version = custom_target(
    'generate-version',
    input: '_build_utils/gitversion.py',
    output: 'version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    build_always_stale: true,
    build_by_default: true,
    install: true,
    install_dir: py.get_install_dir() / 'cherab/inversion',
    install_tag: 'python-runtime',
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir: 'cherab/inversion',
  )
endif

subdir('tools')
subdir('tests')
