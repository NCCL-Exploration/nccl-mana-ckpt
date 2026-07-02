#!/bin/bash
gdb -return-child-result -batch \
    -ex "set follow-fork-mode child" \
    -ex "break exit" \
    -ex "break _exit" \
    -ex "break exit_group" \
    -ex "run" \
    -ex "echo \n\n--- EXIT CALLED, BACKTRACE: ---\n\n" \
    -ex "thread apply all bt full" \
    -ex "quit" \
    --args "$@"
