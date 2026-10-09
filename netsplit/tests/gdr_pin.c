/*
 * gdr_pin: does GDRCopy pin GPU memory after a CUDA checkpoint and restore?
 * libfabric pins every CUDA registration with GDRCopy and copies small
 * messages into GPU memory through the mapping; without GDRCopy it falls back
 * to cudaMemcpy + cudaDeviceSynchronize, which deadlocks against NCCL's
 * kernels.  After a restore, with the handle opened before the checkpoint
 * (what a resumed process has) and with a new handle (what a restarted
 * netsplit lower half would have), pins A (allocated before) and B (after),
 * then writes through the mapping and reads back with cudaMemcpy.
 *
 *   gdr_pin -d DIR      (gdr_probe.sh's driver toggles the pid in DIR/ready)
 */
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <gdrapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SIZE (8UL << 20)

static void pin(gdr_t g, const char *phase, const char *what, void *p) {
    gdr_mh_t mh;
    void *map = NULL;
    int rc = gdr_pin_buffer(g, (unsigned long)p, SIZE, 0, 0, &mh);
    if (rc != 0) {
        printf("PIN  %-7s %-14s pin failed: %d (%s)\n", phase, what, rc, strerror(rc));
        return;
    }
    const char *res = "ok";
    rc = gdr_map(g, mh, &map, SIZE);
    if (rc != 0) {
        res = "map failed";
    } else {
        unsigned char src[4096], back[4096];
        memset(src, 0x77, sizeof(src));
        gdr_copy_to_mapping(mh, map, src, sizeof(src));
        if (cudaMemcpy(back, p, sizeof(back), cudaMemcpyDeviceToHost) != cudaSuccess)
            res = "cudaMemcpy failed";
        else if (memcmp(src, back, sizeof(back)) != 0)
            res = "bad data";
        gdr_unmap(g, mh, map, SIZE);
    }
    printf("PIN  %-7s %-14s %s\n", phase, what, res);
    gdr_unpin_buffer(g, mh);
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int c;
    while ((c = getopt(argc, argv, "d:")) != -1)
        if (c == 'd')
            dir = optarg;
    if (dir == NULL)
        return 2;
    setvbuf(stdout, NULL, _IOLBF, 0);
    int dev = getenv("SLURM_LOCALID") ? atoi(getenv("SLURM_LOCALID")) : 0;
    void *a, *b;
    cudaSetDevice(dev);
    if (cudaMalloc(&a, SIZE) != cudaSuccess)
        return 1;
    gdr_t g = gdr_open();
    if (g == NULL) {
        printf("FAIL gdr_open\n");
        return 1;
    }
    pin(g, "before", "A", a);

    char path[4096], tmp[4200];
    snprintf(path, sizeof(path), "%s/ready", dir);
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *fp = fopen(tmp, "w");
    fprintf(fp, "%d\n", (int)getpid());
    fclose(fp);
    rename(tmp, path);
    snprintf(path, sizeof(path), "%s/restored", dir);
    while (access(path, F_OK) != 0)
        usleep(100000);

    cudaMalloc(&b, SIZE);
    pin(g, "after", "A old-handle", a);
    pin(g, "after", "B old-handle", b);
    gdr_t g2 = gdr_open();
    pin(g2, "after", "A new-handle", a);
    pin(g2, "after", "B new-handle", b);
    gdr_close(g2);
    gdr_close(g);
    printf("RESULT gdr_pin: done\n");
    return 0;
}
