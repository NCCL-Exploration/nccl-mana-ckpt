/*
 * Regression test for the plugin's real-symbol resolution.
 *
 *   ./nccl_dlopen_test
 *
 * Every other test links -lnccl into the test binary, so libnccl is DT_NEEDED on
 * the main executable and lands in the global scope, where the plugin's
 * dlsym(RTLD_NEXT, ...) finds it. A process that reaches NCCL only through an
 * RTLD_LOCAL dlopen -- as CPython does for libtorch_cuda.so -- keeps it out of
 * that scope. Interposition still works, so the plugin gets called and then
 * cannot call through:
 *
 *   [NCCL PLUGIN] dlsym(RTLD_NEXT, "ncclGroupStart") returned NULL
 *   ncclGroupStart -> 3 (internal error)
 *
 * which is how the PyTorch DDP example failed, with no checkpoint involved.
 * ncclGroupStart is where it surfaces because ncclGetUniqueId is not interposed,
 * making it the first wrapped symbol a PyTorch job calls.
 *
 * This executable must NOT be linked against NCCL -- that is the whole point.
 */
#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *h;
    int (*fn)(void);
    int rc;

    printf("[app] NCCL reached only via dlopen(RTLD_LOCAL); the plugin cannot use RTLD_NEXT\n");
    fflush(stdout);

    h = dlopen("./libnccl_dlopen_helper.so", RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) {
        printf("FAIL dlopen: %s\n", dlerror());
        return 2;
    }
    *(void **)&fn = dlsym(h, "run_pytorch_sequence");
    if (fn == NULL) {
        printf("FAIL dlsym: %s\n", dlerror());
        return 2;
    }
    rc = fn();
    if (rc != 0)
        return rc;

    printf("PASS dlopen: every interposed symbol resolved without RTLD_NEXT\n");
    fflush(stdout);
    return 0;
}
