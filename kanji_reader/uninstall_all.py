import subprocess

def get_installed_packages():
    # Get the list of installed packages from pip freeze
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    packages = result.stdout.decode('utf-8').split('\n')
    # Extract package names (ignore versions and empty lines)
    packages = [pkg.split('==')[0] for pkg in packages if '==' in pkg]
    return packages

def uninstall_packages(packages):
    # Packages to exclude from uninstallation
    exclude_packages = [
        'pip',
        'application-utility',
        'argcomplete',
        'attrs',
        'beautifulsoup4',
        'Brlapi',
        'btrfsutil',
        'cachetools',
        'certifi',
        'cffi',
        'charset-normalizer',
        'click',
        'cryptography',
        'cupshelpers',
        'dbus-python',
        'distlib',
        'distro',
        'docopt',
        'filelock',
        'idna',
        'inputs',
        'keyutils',
        'libfdt',
        'libvirt-python',
        'lit',
        'louis',
        'lxml',
        'moddb',
        'netsnmp-python',
        'nftables',
        'npyscreen',
        'packaging',
        'pacman_mirrors',
        'pillow',
        'pipx',
        'platformdirs',
        'ProtonUp-Qt',
        'psutil',
        'pwquality',
        'pyaml',
        'pycairo',
        'pycparser',
        'pycryptodomex',
        'pycups',
        'PyGObject',
        'PyQt5',
        'PyQt5_sip',
        'PySide6',
        'pysmbc',
        'pyxdg',
        'PyYAML',
        'reportlab',
        'requests',
        'shiboken6',
        'shiboken6-generator',
        'six',
        'smbus',
        'soupsieve',
        'steam',
        'TBB',
        'udiskie',
        'urllib3',
        'userpath',
        'vdf',
        'virtualenv',
        'wheel',
        'zstandard'
    ]

    for package in packages:
        if package not in exclude_packages:
            subprocess.check_call(['pip', 'uninstall', package, '-y'])

if __name__ == "__main__":
    packages = get_installed_packages()
    uninstall_packages(packages)
