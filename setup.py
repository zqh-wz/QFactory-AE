import setuptools
import subprocess

if __name__ == '__main__':
    # Get version from git commit
    try:
        revision = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        revision = 'unknown-version'

    setuptools.setup(
        name='qfactory',
        version=f"0.0.1+{revision}",
        packages=[
            'qfactory',
            'qfactory/core',
            'qfactory/jit',
        ],
        package_data={
            'qfactory': [
                'include/**/*'
            ]
        }
    )
