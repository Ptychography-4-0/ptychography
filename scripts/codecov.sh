#!/bin/bash

set -eu

BIN=codecov

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    URL=https://uploader.codecov.io/v0.1.17/linux/codecov
    SHA='ca88335829e3a5b589674a200fdd1dae8f2ef27775647bc3aef6677266a6fda2  codecov'
elif [[ "$OSTYPE" == "darwin" ]]; then
    URL='https://uploader.codecov.io/v0.3.2/macos/codecov'
    SHA='ccc032e70958ea3eca9bd15c7fdad5bbacc279c3bab22f227417573356569666  codecov'
elif [[ "$OSTYPE" == "msys" ]]; then
    URL='https://uploader.codecov.io/v0.3.2/windows/codecov.exe'
    SHA='9dce1ac2d35f550d96d46f8a1116b4f03740d61ca4508bfccdbf1ea1dfeabe4d  codecov.exe'
    BIN=codecov.exe
else
    echo "unknown OSTYPE: $OSTYPE";
    exit 1;
fi

# pin the versions so we can compare sha
curl -Os "$URL"
echo "$SHA" | shasum -b -a 256 -c
chmod +x "$BIN"
./$BIN "$@"
