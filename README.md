# Quick Start

## Configure and Make DMTCP

```sh
echo "cd dmtcp && ./configure && make" | bash
```

## Test Plugin

```sh
make
```

## Run Tests

```sh
cd tests
```

### Checkpoint

Checkpoint the test executable. This should make the plugin first if it does not exist.

```sh
make checkpoint
```

### Restart

Restart from the latest checkpoint file.

```sh
make restart
```

### Quit

Kill coordinator and running process.

```sh
make quit
```
