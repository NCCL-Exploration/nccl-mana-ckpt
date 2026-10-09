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
 * It also reports which way gdrdrv pins (persistent or traditional P2P
 * mappings), and tries the CPU mapping that could stand in for GDRCopy: an
 * mmap() of the buffer's dma-buf (cuMemGetHandleForAddressRange), for a
 * cudaMalloc buffer and a cuMemCreate one, before and after the restore.
 *
 *   PIN  <phase> <buffer> ok|<error>
 *   MAP  <phase> <buffer> ok|bad data|<error>
 *
 *   gdr_pin -d DIR [-m keep|closed|fresh]   (gdr_pin.sh toggles the pid in DIR/ready)
 *
 * -m says what GDRCopy state crosses the checkpoint. Every pin is released
 * (gdr_unpin_buffer) right after its test, so no pin ever does.
 *   keep    the GDRCopy handle (an open /dev/gdrdrv) stays open, as in a
 *           process whose libfabric keeps it (the default);
 *   closed  the handle is closed before the checkpoint and a new one opened
 *           after the restore, as a netsplit lower half could do at resume;
 *   fresh   GDRCopy is first opened after the restore, as in a restarted
 *           netsplit lower half.
 * -t asks the CUDA driver for each buffer's P2P tokens
 * (cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_P2P_TOKENS)) after the restore,
 * before pinning. That legacy call registers the memory for third-party
 * access, which nvidia_p2p_get_pages*() look up by process when given no
 * token (src/nvidia/src/kernel/gpu/bus/p2p.c, RmP2PGetInfoWithoutToken).
 */

#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <gdrapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define SIZE (8UL << 20)

/* The legacy token query, for -t. */
static void tokens(const char *what, void *p) {
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tok;
    CUresult r = cuPointerGetAttribute(&tok, CU_POINTER_ATTRIBUTE_P2P_TOKENS, (CUdeviceptr)p);
    printf("TOK  after   %-14s %s (p2pToken %llx vaSpaceToken %x)\n", what,
           r == CUDA_SUCCESS ? "ok" : "failed", r == CUDA_SUCCESS ? tok.p2pToken : 0ULL,
           r == CUDA_SUCCESS ? tok.vaSpaceToken : 0);
    if (r != CUDA_SUCCESS)
        printf("TOK  after   %-14s CUresult %d\n", what, (int)r);
}

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

/* CPU writes into GPU memory through an mmap() of its dma-buf. */
static void dmabuf_map(const char *phase, const char *what, void *p) {
    int fd = -1;
    CUresult cr = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)p, SIZE,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (cr != CUDA_SUCCESS) {
        printf("MAP  %-7s %-14s no dma-buf fd (cuMemGetHandleForAddressRange %d)\n", phase, what,
               (int)cr);
        return;
    }
    void *map = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        int e = errno;
        /* Read-only, to tell "no mmap at all" from "no writable mmap". */
        void *ro = mmap(NULL, SIZE, PROT_READ, MAP_SHARED, fd, 0);
        printf("MAP  %-7s %-14s mmap failed: %s (read-only mmap: %s)\n", phase, what, strerror(e),
               ro == MAP_FAILED ? strerror(errno) : "ok");
        if (ro != MAP_FAILED)
            munmap(ro, SIZE);
        close(fd);
        return;
    }
    unsigned char src[4096], back[4096];
    memset(src, 0x5c, sizeof(src));
    memcpy(map, src, sizeof(src));
    __sync_synchronize();
    munmap(map, SIZE);
    close(fd);
    const char *res = "ok";
    if (cudaMemcpy(back, p, sizeof(back), cudaMemcpyDeviceToHost) != cudaSuccess)
        res = "cudaMemcpy failed";
    else if (memcmp(src, back, sizeof(back)) != 0)
        res = "bad data";
    printf("MAP  %-7s %-14s %s\n", phase, what, res);
}

/* cuMemCreate memory, as NCCL allocates its buffers. */
static void *vmm_alloc(int dev) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    CUmemGenericAllocationHandle h;
    CUdeviceptr va;
    if (cuMemCreate(&h, SIZE, &prop, 0) != CUDA_SUCCESS ||
        cuMemAddressReserve(&va, SIZE, 0, 0, 0) != CUDA_SUCCESS ||
        cuMemMap(va, SIZE, 0, h, 0) != CUDA_SUCCESS)
        return NULL;
    CUmemAccessDesc ad = {.location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = dev},
                          .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    if (cuMemSetAccess(va, SIZE, &ad, 1) != CUDA_SUCCESS)
        return NULL;
    return (void *)va;
}

int main(int argc, char **argv) {
    const char *dir = NULL, *mode = "keep";
    int c, want_tokens = 0;
    while ((c = getopt(argc, argv, "d:m:t")) != -1) {
        if (c == 'd')
            dir = optarg;
        else if (c == 'm')
            mode = optarg;
        else if (c == 't')
            want_tokens = 1;
    }
    if (dir == NULL ||
        (strcmp(mode, "keep") != 0 && strcmp(mode, "closed") != 0 && strcmp(mode, "fresh") != 0))
        return 2;
    setvbuf(stdout, NULL, _IOLBF, 0);
    int dev = getenv("SLURM_LOCALID") ? atoi(getenv("SLURM_LOCALID")) : 0;
    void *a, *b;
    cudaSetDevice(dev);
    if (cudaMalloc(&a, SIZE) != cudaSuccess)
        return 1;
    printf("INFO mode %s\n", mode);
    gdr_t g = NULL;
    void *v = vmm_alloc(dev);
    if (strcmp(mode, "fresh") != 0) {
        g = gdr_open();
        if (g == NULL) {
            printf("FAIL gdr_open\n");
            return 1;
        }
        int rmaj = 0, rmin = 0, dmaj = 0, dmin = 0, persistent = -1, force_pcie = -1;
        gdr_runtime_get_version(&rmaj, &rmin);
        gdr_driver_get_version(g, &dmaj, &dmin);
        gdr_get_attribute(g, GDR_ATTR_USE_PERSISTENT_MAPPING, &persistent);
        gdr_get_attribute(g, GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE, &force_pcie);
        printf("INFO libgdrapi %d.%d, gdrdrv %d.%d, persistent mapping %d, force-PCIe pin %d\n",
               rmaj, rmin, dmaj, dmin, persistent, force_pcie);
        pin(g, "before", "A", a);
        if (v != NULL)
            pin(g, "before", "V cuMemCreate", v);
    }
    if (strcmp(mode, "keep") == 0) {
        dmabuf_map("before", "A cudaMalloc", a);
        if (v != NULL)
            dmabuf_map("before", "V cuMemCreate", v);
    }
    if (strcmp(mode, "closed") == 0) {
        gdr_close(g);
        g = NULL;
    }

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
    void *w = vmm_alloc(dev);
    if (want_tokens) {
        tokens("A", a);
        tokens("B", b);
        if (v != NULL)
            tokens("V cuMemCreate", v);
        if (w != NULL)
            tokens("W cuMemCreate", w);
    }
    if (g != NULL) {
        pin(g, "after", "A old-handle", a);
        pin(g, "after", "B old-handle", b);
    }
    gdr_t g2 = gdr_open();
    if (g2 == NULL) {
        printf("FAIL gdr_open after the restore\n");
        return 1;
    }
    pin(g2, "after", "A new-handle", a);
    pin(g2, "after", "B new-handle", b);
    if (v != NULL)
        pin(g2, "after", "V new-handle", v);
    if (w != NULL)
        pin(g2, "after", "W new-handle", w);
    if (strcmp(mode, "keep") == 0) {
        dmabuf_map("after", "A cudaMalloc", a);
        dmabuf_map("after", "B cudaMalloc", b);
        if (v != NULL)
            dmabuf_map("after", "V cuMemCreate", v);
        if (w != NULL)
            dmabuf_map("after", "W cuMemCreate", w);
    }
    gdr_close(g2);
    if (g != NULL)
        gdr_close(g);
    printf("RESULT gdr_pin: done\n");
    return 0;
}
