py.install_sources(
  [
    '__init__.py',
    'derivative.py',
    'core.py',
    'gcv.py',
    'lcurve.py',
    'mfr.py',
  ],
  subdir: 'cherab/inversion',
)

# Copy the main __init__.py files to the build dir
__init__py = fs.copyfile('__init__.py')

# Generate version.py for sdist
meson.add_dist_script(
  ['_build_utils/gitversion.py', '--meson-dist', '--write', 'cherab/inversion/version.py']
)
if not fs.exists('cherab/inversion/version.py')
  custom_target(
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

# subdir('bayesian')
subdir('tools')
subdir('data')
subdir('tests')
