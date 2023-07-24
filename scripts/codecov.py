#!/usr/bin/env python
import sys
import hashlib
import platform
import subprocess
from urllib.request import urlretrieve


s = platform.system()

if s == "Linux":
    url = 'https://uploader.codecov.io/v0.6.1/linux/codecov'
    sha = '0c9b79119b0d8dbe7aaf460dc3bd7c3094ceda06e5ae32b0d11a8ff56e2cc5c5'
    fname = './codecov'
elif s == 'Darwin':
    url = 'https://uploader.codecov.io/v0.6.1/macos/codecov'
    sha = '62ba56f0f0d62b28e955fcfd4a3524c7c327fcf8f5fcb5124cccf88db358282e'
    fname = './codecov'
elif s == 'Windows':
    url = 'https://uploader.codecov.io/v0.6.1/windows/codecov.exe'
    sha = '6b95584fbb252b721b73ddfe970d715628879543d119f1d2ed08b073155f7d06'
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
