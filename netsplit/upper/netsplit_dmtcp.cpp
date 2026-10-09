/*
 * netsplit's DMTCP plugin, in the upper half: what MANA's mpi_plugin.cpp and
 * uh_wrappers.cpp do for the split itself, without MPI.
 *
 *  - lh_info: found at launch (NETSPLIT_LH_INFO_ADDR) and again at restart,
 *    from the file of the new lower half.
 *  - mmap/munmap: the upper half's mappings go to the lower half's arena,
 *    which records them, and dmtcp_skip_memory_region_ckpting() saves only
 *    those: the lower half's memory stays out of the image.
 *  - Signal handlers run on the upper half's TLS, also when the signal comes
 *    while the thread runs back-end code on a borrowed lower-half TLS.
 *  - Per-rank checkpoint directories (ckpt_rank_<rank>, as the lower half
 *    looks for them at restart), the fd numbers to keep free at restart
 *    (header.netsplit), and closing the leftover placeholders after it.
 *  - DMTCP file records of fds that became anonymous inodes (MANA db5b61b).
 *  - The proxy's checks and its new epoch at restart.
 *
 * Plugin order: --with-plugin libnccl-net-netsplit.so:libncclplugin.so:
 * libdmtcp_cuda.so.  Forward events (PRESUSPEND, PRECHECKPOINT, RESTART) run
 * in that order: this plugin stops locking in mmap() before the NCCL plugin
 * tears communicators down, and switches to the new lower half before the
 * NCCL plugin's restart.  RUNNING_AFTER runs in reverse: CUDA's restore, then
 * the NCCL rebuild, which reaches the new back end through the proxy.
 */
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <vector>

#include "dmtcp.h"
#include "jfilesystem.h"
#include "procmapsarea.h"

#include "netsplit-api.h"
#include "netsplit-uh.h"

extern "C" pid_t dmtcp_get_real_pid() __attribute((weak));

NetsplitInfo_t *lh_info = NULL;

typedef std::vector<MmapInfo_t> &(*get_mmapped_list_fptr_t)(int *num);
static std::vector<MmapInfo_t> uh_mmaps;
static char *saved_uh_stack_start, *saved_uh_stack_end;

/* 1 from PRESUSPEND until the process runs again: mmap() then takes no
 * DMTCP lock (the checkpoint thread calls it, e.g. in NCCL's teardown). */
static volatile int g_in_ckpt = 0;

/* ---- lh_info ---------------------------------------------------------------- */

static int read_lh_info_file(void) {
    char host[256], path[512];
    if (gethostname(host, sizeof(host)) != 0)
        return -1;
    host[sizeof(host) - 1] = '\0';
    long pid = dmtcp_get_real_pid != NULL ? (long)dmtcp_get_real_pid() : (long)getpid();
    netsplit_lh_info_path(path, sizeof(path), host, pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    NetsplitInfo_t *p = NULL;
    ssize_t rc = read(fd, &p, sizeof(p));
    close(fd);
    unlink(path);
    if (rc != (ssize_t)sizeof(p) || p == NULL)
        return -1;
    lh_info = p;
    return 0;
}

extern "C" int netsplit_lh_info_init(int restart) {
    /* At restart the environment is the checkpointed one: NETSPLIT_LH_INFO_ADDR
     * names the old lower half, so the new one's file comes first. */
    if (restart && read_lh_info_file() == 0)
        return 0;
    const char *s = getenv(NETSPLIT_LH_INFO_ENV);
    if (s != NULL && *s != '\0') {
        lh_info = (NetsplitInfo_t *)strtoul(s, NULL, 16);
    } else if (read_lh_info_file() != 0) {
        return -1;
    }
    if (lh_info != NULL && lh_info->version != NETSPLIT_API_VERSION) {
        fprintf(stderr, "[netsplit] lower half API version %u, upper half %u\n",
                lh_info->version, NETSPLIT_API_VERSION);
        lh_info = NULL;
        return -1;
    }
    return lh_info != NULL ? 0 : -1;
}

/* ---- mmap/munmap ------------------------------------------------------------- */

/* The upper half's mappings go through the lower half's arena allocator
 * (mem-wrapper.cpp), which records them.  It switches FS itself. */
extern "C" void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (lh_info == NULL && netsplit_lh_info_init(0) != 0) {
        /* Not under netsplit-lh: the next mmap, normally libc's. */
        static void *(*next)(void *, size_t, int, int, int, off_t) = NULL;
        if (next == NULL)
            *(void **)&next = dlsym(RTLD_NEXT, "mmap");
        return next(addr, length, prot, flags, fd, offset);
    }
    void *(*lh_mmap)(void *, size_t, int, int, int, off_t) =
        (void *(*)(void *, size_t, int, int, int, off_t))lh_info->mmap;
    int lock = !g_in_ckpt;
    if (lock)
        DMTCP_PLUGIN_DISABLE_CKPT();
    void *ret = lh_mmap(addr, length, prot, flags, fd, offset);
    if (lock)
        DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

extern "C" int munmap(void *addr, size_t length) {
    if (lh_info == NULL && netsplit_lh_info_init(0) != 0) {
        static int (*next)(void *, size_t) = NULL;
        if (next == NULL)
            *(void **)&next = dlsym(RTLD_NEXT, "munmap");
        return next(addr, length);
    }
    int (*lh_munmap)(void *, size_t) = (int (*)(void *, size_t))lh_info->munmap;
    int lock = !g_in_ckpt;
    if (lock)
        DMTCP_PLUGIN_DISABLE_CKPT();
    int ret = lh_munmap(addr, length);
    if (lock)
        DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

/* ---- memory regions saved ------------------------------------------------------ */

static inline int region_contains(const void *hs, const void *he, const void *ns,
                                  const void *ne) {
    return ns >= hs && ne <= he;
}

/* Regions only the lower half uses (from MANA's isLhDevice()).  "/dev/zero"
 * is ambiguous (every MAP_SHARED|MAP_ANONYMOUS region has that name, the CUDA
 * checkpoint's staging buffers included) and is decided by uh_mmaps.  NCCL's
 * /dev/shm segments are gone before the checkpoint (teardown). */
static bool is_lh_device(const ProcMapsArea *area) {
    return strstr(area->name, "/dev/kgni") || strstr(area->name, "/dev/xpmem") ||
           strstr(area->name, "/dev/shm") || strstr(area->name, "/dev/cxi");
}

/* Saves a region only if the upper half mapped it (uh_mmaps), clipped to
 * that mapping; the caller (writeckpt.cpp) comes back for the rest of a
 * kernel-merged area. */
extern "C" int dmtcp_skip_memory_region_ckpting(ProcMapsArea *area) {
    if (lh_info == NULL)
        return 0;
    if (is_lh_device(area))
        return 1;
    if (strcmp(area->name, "[heap]") == 0)
        return 1; /* the lower half's heap; the upper half's malloc uses arenas */
    if (strstr(area->name, "/anon_hugepage"))
        return 1;
    if (area->addr >= lh_info->uh_stack_start && area->endAddr <= lh_info->uh_stack_end)
        return 0;
    if (uh_mmaps.empty()) {
        int n;
        uh_mmaps = ((get_mmapped_list_fptr_t)lh_info->mmap_list_fptr)(&n);
    }
    for (MmapInfo_t &r : uh_mmaps) {
        if (region_contains(r.addr, r.addr + r.len, area->addr, area->endAddr))
            return 0;
        if (region_contains(area->addr, area->endAddr, r.addr, r.addr + r.len)) {
            area->addr = r.addr;
            area->endAddr = r.addr + r.len;
            area->size = area->endAddr - area->addr;
            return 0;
        }
        if (area->addr < r.addr && r.addr < area->endAddr && area->endAddr < r.addr + r.len) {
            area->addr = r.addr;
            area->size = area->endAddr - area->addr;
            return 0;
        }
        if (r.addr < area->addr && area->addr < r.addr + r.len && r.addr + r.len < area->endAddr) {
            area->endAddr = r.addr + r.len;
            area->size = area->endAddr - area->addr;
            return 0;
        }
    }
    return 1;
}

/* ---- signals ------------------------------------------------------------------- */

#define MAX_SIGNALS 64
static struct sigaction user_handlers[MAX_SIGNALS];

/* A signal that interrupts back-end code finds the FS of the borrowed
 * lower-half TLS: run the application's handler on the thread's own. */
static void handler_wrapper(int sig) {
    unsigned long fs = ns_getfs(), uh = netproxy_upper_fs_for(fs);
    if (uh != 0)
        ns_setfs(uh);
    user_handlers[sig].sa_handler(sig);
    if (uh != 0)
        ns_setfs(fs);
}

static void sigaction_wrapper(int sig, siginfo_t *info, void *ctx) {
    unsigned long fs = ns_getfs(), uh = netproxy_upper_fs_for(fs);
    if (uh != 0)
        ns_setfs(uh);
    user_handlers[sig].sa_sigaction(sig, info, ctx);
    if (uh != 0)
        ns_setfs(fs);
}

extern "C" int sigaction(int sig, const struct sigaction *act, struct sigaction *old) {
    if (sig <= 0 || sig >= MAX_SIGNALS || sig == dmtcp_get_ckpt_signal())
        return NEXT_FNC(sigaction)(sig, act, old);
    int ret;
    struct sigaction prev = user_handlers[sig];
    if (act != NULL) {
        struct sigaction na = *act;
        if (na.sa_flags & SA_SIGINFO)
            na.sa_sigaction = sigaction_wrapper;
        else if (na.sa_handler != SIG_IGN && na.sa_handler != SIG_DFL)
            na.sa_handler = handler_wrapper;
        ret = NEXT_FNC(sigaction)(sig, &na, NULL);
        if (ret == 0)
            user_handlers[sig] = *act;
    } else {
        ret = NEXT_FNC(sigaction)(sig, NULL, NULL);
    }
    if (old != NULL && ret == 0)
        *old = prev;
    return ret;
}

extern "C" sighandler_t signal(int sig, sighandler_t handler) {
    struct sigaction act, old;
    memset(&act, 0, sizeof(act));
    act.sa_handler = handler;
    act.sa_flags = SA_RESTART;
    sigemptyset(&act.sa_mask);
    if (sigaction(sig, &act, &old) != 0)
        return SIG_ERR;
    return old.sa_handler;
}

static void record_signal_handlers(void) {
    for (int i = 1; i < MAX_SIGNALS; i++)
        NEXT_FNC(sigaction)(i, NULL, &user_handlers[i]);
}

/* ---- checkpoint directory, fds ---------------------------------------------- */

static int launcher_rank(void) {
    const char *vars[] = {"PMI_RANK", "PMIX_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK",
                          "MV2_COMM_WORLD_RANK"};
    for (const char *v : vars) {
        const char *s = getenv(v);
        if (s != NULL && *s != '\0')
            return atoi(s);
    }
    return 0;
}

/* <ckptdir>/ckpt_rank_<rank>, where netsplit-lh looks at restart. */
static void update_ckpt_dir_by_rank(void) {
    const char *dir = dmtcp_get_ckpt_dir();
    dmtcp::string base = dir;
    if (strstr(dir, "ckpt_rank_") != NULL)
        base = jalib::Filesystem::DirName(dir);
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s/ckpt_rank_%d", base.c_str(), launcher_rank());
    dmtcp_set_ckpt_dir(buf);
}

/* The fd numbers open at checkpoint: nccl_restart passes them to the new
 * lower half (NETSPLIT_RESERVED_FDS), which keeps them free for DMTCP. */
static void save_header(void) {
    const char *dir = dmtcp_get_ckpt_dir();
    struct stat st;
    if (stat(dir, &st) == -1)
        mkdir(dir, 0700);
    char path[4096];
    snprintf(path, sizeof(path), "%s/header.netsplit", dir);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
        return;
    dmtcp::string text = "fds=";
    const char *sep = "";
    for (int f : jalib::Filesystem::ListOpenFds()) {
        if (f > 2 && f != fd && !dmtcp_is_protected_fd(f)) {
            char num[16];
            snprintf(num, sizeof(num), "%s%d", sep, f);
            text += num;
            sep = ",";
        }
    }
    text += "\n";
    if (write(fd, text.c_str(), text.size()) < 0) {
        /* Only a hint for the restart; nothing to do. */
    }
    close(fd);
}

/* NETSPLIT_SAVE_MAPS=1: copy /proc/self/maps to <ckpt dir>/maps.<tag>, at the
 * checkpoint and again at restart before the CUDA plugin restores the GPU,
 * to compare the address spaces when cuCheckpointProcessRestore() fails. */
static void save_maps(const char *tag) {
    const char *e = getenv("NETSPLIT_SAVE_MAPS");
    if (e == NULL || strcmp(e, "1") != 0)
        return;
    char path[4096];
    snprintf(path, sizeof(path), "%s/maps.%s", dmtcp_get_ckpt_dir(), tag);
    int in = open("/proc/self/maps", O_RDONLY);
    int out = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    char buf[4096];
    ssize_t n;
    while (in >= 0 && out >= 0 && (n = read(in, buf, sizeof(buf))) > 0) {
        if (write(out, buf, n) != n)
            break;
    }
    if (in >= 0)
        close(in);
    if (out >= 0)
        close(out);
}

static void close_reserved_fds(void) {
    if (lh_info == NULL || lh_info->reserved_fd_ino == 0)
        return;
    for (int fd : jalib::Filesystem::ListOpenFds()) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_dev == lh_info->reserved_fd_dev &&
            st.st_ino == lh_info->reserved_fd_ino)
            close(fd);
    }
    lh_info->reserved_fd_ino = 0;
}

/* DMTCP's file plugin; weak, in case it is not loaded. */
void dmtcp_FileConnList_EventHook(DmtcpEvent_t event, DmtcpEventData_t *data)
    __attribute((weak));

/* An fd that the upper half opened and whose number now names an anonymous
 * inode was closed without DMTCP's wrappers seeing it (the CUDA driver's
 * /dev/nvidiactl, reused for an eventfd): tell the file plugin, and only it,
 * so it does not try to save the "file".  See MANA's db5b61b. */
static void forget_reused_file_fds(void) {
    if (dmtcp_FileConnList_EventHook == NULL)
        return;
    for (int fd : jalib::Filesystem::ListOpenFds()) {
        dmtcp::string device = jalib::Filesystem::GetDeviceName(fd);
        if (device.compare(0, strlen("anon_inode:"), "anon_inode:") == 0) {
            DmtcpEventData_t data;
            data.closeFd.fd = fd;
            dmtcp_FileConnList_EventHook(DMTCP_EVENT_CLOSE_FD, &data);
        }
    }
}

/* ---- NCCL's host-specific state ---------------------------------------------------- */

/* The address of local symbol 'name' in the loaded library whose path
 * contains 'lib', from the file's .symtab (NVIDIA's libnccl keeps it), or
 * NULL. */
struct find_lib {
    const char *lib;
    char path[4096];
    ElfW(Addr) bias;
    int found;
};

static int find_lib_cb(struct dl_phdr_info *info, size_t size, void *arg) {
    (void)size;
    struct find_lib *f = (struct find_lib *)arg;
    if (info->dlpi_name != NULL && strstr(info->dlpi_name, f->lib) != NULL) {
        snprintf(f->path, sizeof(f->path), "%s", info->dlpi_name);
        f->bias = info->dlpi_addr;
        f->found = 1;
        return 1;
    }
    return 0;
}

static void *local_symbol(const char *lib, const char *name) {
    struct find_lib f;
    memset(&f, 0, sizeof(f));
    f.lib = lib;
    dl_iterate_phdr(find_lib_cb, &f);
    if (!f.found)
        return NULL;
    int fd = open(f.path, O_RDONLY);
    if (fd < 0)
        return NULL;
    struct stat st;
    void *addr = NULL;
    if (fstat(fd, &st) == 0) {
        void *map = NULL;
        /* The file only, mapped read-only for the lookup (libc's mmap: this
         * runs at restart, and the mapping is gone right after). */
        static void *(*libc_mmap)(void *, size_t, int, int, int, off_t) = NULL;
        static int (*libc_munmap)(void *, size_t) = NULL;
        if (libc_mmap == NULL) {
            *(void **)&libc_mmap = dlsym(RTLD_NEXT, "mmap");
            *(void **)&libc_munmap = dlsym(RTLD_NEXT, "munmap");
        }
        map = libc_mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map != MAP_FAILED) {
            const ElfW(Ehdr) *eh = (const ElfW(Ehdr) *)map;
            const ElfW(Shdr) *sh = (const ElfW(Shdr) *)((const char *)map + eh->e_shoff);
            for (int i = 0; i < eh->e_shnum && addr == NULL; i++) {
                if (sh[i].sh_type != SHT_SYMTAB)
                    continue;
                const ElfW(Sym) *sym = (const ElfW(Sym) *)((const char *)map + sh[i].sh_offset);
                const char *str = (const char *)map + sh[sh[i].sh_link].sh_offset;
                size_t n = sh[i].sh_size / sizeof(ElfW(Sym));
                for (size_t k = 0; k < n; k++) {
                    if (sym[k].st_name != 0 && strcmp(str + sym[k].st_name, name) == 0) {
                        addr = (void *)(f.bias + sym[k].st_value);
                        break;
                    }
                }
            }
            libc_munmap(map, st.st_size);
        }
    }
    close(fd);
    return addr;
}

/* NCCL computes two host-specific values once per process, and both are in
 * the image:
 *  - bootstrapNetInit() picks the bootstrap interface (bootstrapNetIfAddr,
 *    bootstrapNetIfName) and sets bootstrapNetInitDone.  After a restart on
 *    another node the rebuild's ncclGetUniqueId() binds the old node's
 *    address ("Call to bind failed: Cannot assign requested address").
 *    Only ncclInit()'s call_once calls it, so clearing the flag is not
 *    enough: this calls it again.
 *  - getHostHash() hashes the hostname and boot id under a once flag.  A
 *    stale hash makes NCCL treat ranks on different nodes as one host (P2P,
 *    SHM) or ranks on one node as different hosts.
 * All are file-scope C++ statics, found mangled in libnccl's .symtab.  No
 * thread is inside NCCL at DMTCP_EVENT_RESTART.  NETSPLIT_NCCL_RESET=0 skips
 * this. */
static void *nccl_symbol(const char *name) {
    void *addr = local_symbol("libnccl.so", name);
    if (addr == NULL)
        fprintf(stderr, "[netsplit] restart: %s not found in libnccl's .symtab; a restart "
                        "on another node may fail\n", name);
    return addr;
}

static void reset_nccl_host_state(void) {
    const char *e = getenv("NETSPLIT_NCCL_RESET");
    if (e != NULL && strcmp(e, "0") == 0)
        return;
    int *host_hash_once = (int *)nccl_symbol("_ZZ11getHostHashvE4once");
    if (host_hash_once != NULL)
        *host_hash_once = 0; /* pthread_once_t */
    int *done = (int *)nccl_symbol("_ZL20bootstrapNetInitDone");
    int (*net_init)(void) = (int (*)(void))nccl_symbol("_Z16bootstrapNetInitv");
    if (done == NULL || net_init == NULL)
        return;
    *done = 0;
    int ret = net_init();
    if (ret != 0)
        fprintf(stderr, "[netsplit] restart: NCCL's bootstrapNetInit() failed (%d)\n", ret);
}

/* ---- events ---------------------------------------------------------------------- */

static void netsplit_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data) {
    (void)data;
    switch (event) {
    case DMTCP_EVENT_INIT: {
        if (netsplit_lh_info_init(0) != 0) {
            fprintf(stderr, "[netsplit] not started by netsplit-lh: the network back end "
                            "would be checkpointed with the process\n");
            break;
        }
        record_signal_handlers();
        /* A page just past the heap keeps glibc from growing its main arena
         * there: at restart the break belongs to a new lower half (MANA's
         * DMTCP_EVENT_INIT). */
        void *heap = sbrk(0);
        mmap(heap, 4096, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        break;
    }
    case DMTCP_EVENT_PRESUSPEND:
        g_in_ckpt = 1;
        break;
    case DMTCP_EVENT_PRECHECKPOINT: {
        if (lh_info == NULL)
            break;
        char what[256];
        if (netproxy_live_handles(what, sizeof(what)) != 0)
            fprintf(stderr, "[netsplit] WARNING: back-end handles alive at checkpoint (%s): "
                            "they refer to a lower half that a restart replaces\n", what);
        forget_reused_file_fds();
        uh_mmaps.clear(); /* refetched by dmtcp_skip_memory_region_ckpting() */
        update_ckpt_dir_by_rank();
        save_header();
        save_maps("ckpt");
        saved_uh_stack_start = lh_info->uh_stack_start;
        saved_uh_stack_end = lh_info->uh_stack_end;
        break;
    }
    case DMTCP_EVENT_RESUME:
        g_in_ckpt = 0;
        break;
    case DMTCP_EVENT_RESTART:
        if (netsplit_lh_info_init(1) != 0) {
            fprintf(stderr, "[netsplit] restart: cannot find the new lower half\n");
            abort();
        }
        lh_info->uh_stack_start = saved_uh_stack_start;
        lh_info->uh_stack_end = saved_uh_stack_end;
        uh_mmaps.clear();
        netproxy_restart();
        reset_nccl_host_state();
        save_maps("restart");
        g_in_ckpt = 0;
        break;
    case DMTCP_EVENT_RUNNING:
        close_reserved_fds();
        break;
    default:
        break;
    }
}

DmtcpPluginDescriptor_t netsplit_plugin = {DMTCP_PLUGIN_API_VERSION,
                                           DMTCP_PACKAGE_VERSION,
                                           "netsplit",
                                           "nccl-mana-ckpt",
                                           "xu.yao1@northeastern.edu",
                                           "NCCL's network back end in a lower half",
                                           netsplit_event_hook};

DMTCP_DECL_PLUGIN(netsplit_plugin);
