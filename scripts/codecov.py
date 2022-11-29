#!/usr/bin/env python
import sys
import hashlib
import platform
import subprocess
from urllib.request import urlretrieve


s = platform.system()

if s == "Linux":
    url = 'https://uploader.codecov.io/v0.1.17/linux/codecov'
    sha = 'ca88335829e3a5b589674a200fdd1dae8f2ef27775647bc3aef6677266a6fda2'
    fname = './codecov'
elif s == 'Darwin':
    url = 'https://uploader.codecov.io/v0.3.2/macos/codecov'
    sha = 'ccc032e70958ea3eca9bd15c7fdad5bbacc279c3bab22f227417573356569666'
    fname = './codecov'
elif s == 'Windows':
    url = 'https://uploader.codecov.io/v0.3.2/windows/codecov.exe'
    sha = '9dce1ac2d35f550d96d46f8a1116b4f03740d61ca4508bfccdbf1ea1dfeabe4d'
    fname = 'codecov.exe'
else:
    print(f"unknown system: {s}")
    sys.exit(1)

if __name__ == "__main__":
    urlretrieve(url, fname)

    with open(fname, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()

    if h != sha:
        print(f"invalid sha256sum: expected {sha}, got {h}")
        sys.exit(1)

    subprocess.run(["chmod", "+x", fname], check=False)
    subprocess.run([fname] + sys.argv[1:], check=True)
