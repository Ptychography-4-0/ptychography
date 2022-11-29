#!/bin/bash

set -eu

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    URL=https://uploader.codecov.io/v0.1.17/linux/codecov
    SHA='ca88335829e3a5b589674a200fdd1dae8f2ef27775647bc3aef6677266a6fda2'
elif [[ "$OSTYPE" == "darwin" ]]; then
    URL='https://uploader.codecov.io/v0.3.2/macos/codecov'
    SHA='ccc032e70958ea3eca9bd15c7fdad5bbacc279c3bab22f227417573356569666'
else
    echo "unknown OSTYPE: $OSTYPE";
    exit 1;
fi

# pin the versions so we can compare sha
curl -Os "$URL"
echo "$SHA  codecov" | shasum -b -a 256 -c
chmod +x codecov
./codecov "$@"
