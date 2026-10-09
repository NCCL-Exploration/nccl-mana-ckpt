/*
 * The lower half's side of netsplit: NCCL's network back end (aws-ofi-nccl)
 * and the CUDA stub it links with, a logger for the back end, and the pool of
 * parked threads whose TLS the upper half's net proxy borrows.
 *
 * Everything here runs on a lower-half TLS: backend_load(), tls_acquire()
 * and tls_release() on the lower half's main TLS (lh_info->fsaddr), with the
 * upper half's lock held; the logger on whatever borrowed TLS the back end
 * runs on.  None of it calls upper-half code.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <asm/prctl.h>
#include <dlfcn.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "backend.h"

// ------------------------------------------------------------------ logger

// NCCL_DEBUG as NCCL reads it: WARN unless it says otherwise.
static int
log_level()
{
  static int level = -1;
  if (level < 0) {
    const char *v = getenv("NCCL_DEBUG");
    level = NCCL_LOG_WARN;
    if (v != NULL) {
      if (strcasecmp(v, "NONE") == 0) {
        level = NCCL_LOG_NONE;
      } else if (strcasecmp(v, "VERSION") == 0) {
        level = NCCL_LOG_VERSION;
      } else if (strcasecmp(v, "INFO") == 0) {
        level = NCCL_LOG_INFO;
      } else if (strcasecmp(v, "TRACE") == 0) {
        level = NCCL_LOG_TRACE;
      }
    }
  }
  return level;
}

// The back end's logger.  NCCL's own logger is upper-half code, which must
// not run on a lower-half TLS, so the back end gets this one.  One write()
// per message keeps the lines of concurrent threads apart.
static void
lh_logger(ncclDebugLogLevel level, unsigned long flags, const char *file,
          int line, const char *fmt, ...)
{
  (void)flags;
  if (level == NCCL_LOG_NONE ||
      (level != NCCL_LOG_ABORT && (int)level > log_level())) {
    return;
  }
  char buf[1024];
  int n = snprintf(buf, sizeof(buf), "[netsplit-lh %d] %s:%d ",
                   (int)syscall(SYS_gettid), file ? file : "?", line);
  if (n < 0 || n >= (int)sizeof(buf)) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  int m = vsnprintf(buf + n, sizeof(buf) - n - 1, fmt, ap);
  va_end(ap);
  if (m < 0) {
    return;
  }
  n += m < (int)(sizeof(buf) - n - 1) ? m : (int)(sizeof(buf) - n - 2);
  buf[n++] = '\n';
  ssize_t rc = write(2, buf, n);
  (void)rc;
}

// ---------------------------------------------------------------- back end

static void *g_stub = NULL;
static void *g_backend = NULL;
static char g_error[1024];

// The default CUDA stub: ../lib/libcudart_stub.so next to this program's
// bin directory, else in that directory itself.
static int
default_stub_path(char *buf, size_t len)
{
  char exe[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n <= 0) {
    return -1;
  }
  exe[n] = '\0';
  char *slash = strrchr(exe, '/');
  if (slash == NULL) {
    return -1;
  }
  *slash = '\0';
  const char *candidates[] = { "/../lib/libcudart_stub.so",
                               "/libcudart_stub.so" };
  for (const char *c : candidates) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s%s", exe, c);
    if (realpath(path, buf) != NULL && access(buf, R_OK) == 0) {
      (void)len;
      return 0;
    }
  }
  return -1;
}

static int
fail(const char *what, const char *detail)
{
  snprintf(g_error, sizeof(g_error), "%s: %s", what,
           detail != NULL ? detail : "unknown error");
  lh_info->backend_error = g_error;
  return -1;
}

// Loads the CUDA stub, then the back end.  The stub goes first and
// RTLD_GLOBAL: aws-ofi-nccl has DT_NEEDED libcudart.so.13 with versioned
// symbols, which then resolve to the stub's (same SONAME and version node).
// libfabric would otherwise dlopen() the real CUDA libraries for its device
// memory support and start a second CUDA instance in the process; netsplit
// stages network traffic through host memory, so FI_HMEM=system keeps it to
// host memory unless the user set FI_HMEM.  Idempotent.
static int
backend_load(void)
{
  if (lh_info->net != NULL) {
    return 0;
  }
  setenv("FI_HMEM", "system", 0);

  if (g_stub == NULL) {
    char stub[PATH_MAX];
    const char *env = getenv("NETSPLIT_STUB");
    if (env != NULL && env[0] != '\0') {
      snprintf(stub, sizeof(stub), "%s", env);
    } else if (default_stub_path(stub, sizeof(stub)) != 0) {
      return fail("no CUDA stub", "set NETSPLIT_STUB to libcudart_stub.so");
    }
    g_stub = dlopen(stub, RTLD_NOW | RTLD_GLOBAL);
    if (g_stub == NULL) {
      return fail("cannot load the CUDA stub", dlerror());
    }
    lh_info->stub_set_ndev =
      (void (*)(int))dlsym(g_stub, "netsplit_stub_set_ndev");
    lh_info->stub_set_thread_device =
      (void (*)(int))dlsym(g_stub, "netsplit_stub_set_thread_device");
    lh_info->stub_add_device_range =
      (int (*)(void *, size_t, int))dlsym(g_stub,
                                         "netsplit_stub_add_device_range");
    lh_info->stub_remove_device_range =
      (void (*)(void *))dlsym(g_stub, "netsplit_stub_remove_device_range");
    if (lh_info->stub_set_ndev == NULL ||
        lh_info->stub_set_thread_device == NULL) {
      return fail("the CUDA stub lacks netsplit_stub_*", stub);
    }
  }

  if (g_backend == NULL) {
    char backend[PATH_MAX];
    const char *env = getenv("NETSPLIT_BACKEND");
    const char *home = getenv("NCCL_HOME");
    if (env != NULL && env[0] != '\0') {
      snprintf(backend, sizeof(backend), "%s", env);
    } else if (home != NULL && home[0] != '\0') {
      snprintf(backend, sizeof(backend), "%s/plugin/lib/libnccl-net.so", home);
    } else {
      snprintf(backend, sizeof(backend), "libnccl-net.so");
    }
    g_backend = dlopen(backend, RTLD_NOW | RTLD_LOCAL);
    if (g_backend == NULL) {
      return fail("cannot load the network back end", dlerror());
    }
  }
  ncclNet_v6_t *net = (ncclNet_v6_t *)dlsym(g_backend, "ncclNetPlugin_v6");
  if (net == NULL) {
    return fail("the network back end has no ncclNetPlugin_v6", dlerror());
  }
  lh_info->backend_error = NULL;
  lh_info->net = net;
  return 0;
}

// ---------------------------------------------------------------- TLS pool

// The TLS of a parked thread lets one upper-half thread run lower-half code
// as if it were a lower-half thread: glibc's per-thread state (errno, stdio
// and malloc caches, pthread_self()) is that of the parked thread, which
// itself never runs again.  An upper-half thread keeps its TLS until it
// exits, so no two threads ever share one.
#define TLS_MAX 1024
#define PARKED_STACK_SIZE (256 * 1024)

static unsigned long g_free_fs[TLS_MAX];
static int g_nfree = 0;
static volatile int g_tls_lock = 0;

static void
tls_lock()
{
  while (__sync_lock_test_and_set(&g_tls_lock, 1)) {
    sched_yield();
  }
}

static void
tls_unlock()
{
  __sync_lock_release(&g_tls_lock);
}

// Reports its FS, then waits forever with all signals blocked (they are
// blocked before it is created, and it inherits the mask).
static void *
parked_main(void *arg)
{
  unsigned long fs = 0;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &fs);
  __atomic_store_n((unsigned long *)arg, fs, __ATOMIC_RELEASE);
  int never = 0;
  for (;;) {
    syscall(SYS_futex, &never, FUTEX_WAIT_PRIVATE, 0, NULL, NULL, 0);
  }
  return NULL;
}

static int
park_new_thread(unsigned long *fs)
{
  static unsigned long reported;
  __atomic_store_n(&reported, 0, __ATOMIC_RELEASE);
  sigset_t all, old;
  sigfillset(&all);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, PARKED_STACK_SIZE);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  pthread_t thread;
  int rc = pthread_create(&thread, &attr, parked_main, &reported);
  pthread_sigmask(SIG_SETMASK, &old, NULL);
  pthread_attr_destroy(&attr);
  if (rc != 0) {
    return -1;
  }
  unsigned long value;
  while ((value = __atomic_load_n(&reported, __ATOMIC_ACQUIRE)) == 0) {
    sched_yield();
  }
  *fs = value;
  lh_info->tls_parked++;
  return 0;
}

static int
tls_acquire(unsigned long *fs)
{
  int rc = 0;
  tls_lock();
  if (g_nfree > 0) {
    *fs = g_free_fs[--g_nfree];
  } else if (lh_info->tls_parked >= TLS_MAX) {
    rc = -1;
  } else {
    rc = park_new_thread(fs);
  }
  if (rc == 0) {
    lh_info->tls_in_use++;
  }
  tls_unlock();
  return rc;
}

static void
tls_release(unsigned long fs)
{
  tls_lock();
  if (fs != 0 && g_nfree < TLS_MAX) {
    g_free_fs[g_nfree++] = fs;
    lh_info->tls_in_use--;
  }
  tls_unlock();
}

// ------------------------------------------------------------------ setup

void
netsplit_init_backend_info(LowerHalfInfo_t *info)
{
  info->backend_load = &backend_load;
  info->backend_error = NULL;
  info->net = NULL;
  info->logger = &lh_logger;
  info->tls_acquire = &tls_acquire;
  info->tls_release = &tls_release;
  info->tls_parked = 0;
  info->tls_in_use = 0;
  info->stub_set_ndev = NULL;
  info->stub_set_thread_device = NULL;
  info->stub_add_device_range = NULL;
  info->stub_remove_device_range = NULL;
}
