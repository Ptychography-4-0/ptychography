# Requirements

[Requirements](../Requirements.md)

# Install

## Linux

```bash
  mkdir build && cd build
  # maybe you need to set: -DPYTHON_EXECUTABLE:FILEPATH=
  cmake ..
  cmake --build .
```

## Windows

```bash
  mkdir build; cd build
  cmake -G"Visual Studio 15 2017 Win64" ..
  cmake --build .
```

# Usage

```bash
  python test.py
```
