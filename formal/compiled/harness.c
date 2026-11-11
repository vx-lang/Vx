// The parts of the Vx runtime that a Vx program calls, written for GenMC.
//
// GenMC runs a closed program, and it cannot run the Rust runtime. So this file
// gives the runtime functions that the compiled library calls: tasks, waits and
// wakes. The library code itself (lock, unlock, send, recv, the drop of an Arc)
// is the IR that vxc generated.

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
// GenMC's stdlib.h gives `malloc` and `free` as inline functions, and the Vx IR
// calls them as external functions. So rename the inline ones, and define the
// external ones below.
#define malloc genmc_malloc
#define free genmc_free
#define aligned_alloc genmc_aligned_alloc
#include <stdlib.h>
#undef malloc
#undef free
#undef aligned_alloc

#include <genmc.h>

void *malloc(size_t n) { return genmc_malloc(n); }
void free(void *p) { genmc_free(p); }
void *aligned_alloc(size_t align, size_t n) {
  return genmc_aligned_alloc(align, n);
}

// A `tasks` block. Each `spawn` is a pthread.
#define MAX_TASKS 4

struct group {
  pthread_t threads[MAX_TASKS];
  void *envs[MAX_TASKS];
  int joined[MAX_TASKS];
  int count;
};

struct start {
  void (*run)(void *);
  void *env;
};

static void *task_main(void *arg) {
  struct start *s = arg;
  s->run(s->env);
  free(s);
  return NULL;
}

void *vx_tasks_begin(void) {
  struct group *g = malloc(sizeof(struct group));
  g->count = 0;
  for (int i = 0; i < MAX_TASKS; i++)
    g->joined[i] = 0;
  return g;
}

long vx_task_start_at(struct group *g, void (*run)(void *), void *env,
                      const char *file, long line) {
  assert(g->count < MAX_TASKS);
  struct start *s = malloc(sizeof(struct start));
  s->run = run;
  s->env = env;
  g->envs[g->count] = env;
  pthread_create(&g->threads[g->count], NULL, task_main, s);
  return g->count++;
}

long vx_task_start(struct group *g, void (*run)(void *), void *env) {
  return vx_task_start_at(g, run, env, NULL, 0);
}

void vx_task_join(struct group *g, long i) {
  if (g == NULL || g->joined[i])
    return;
  pthread_join(g->threads[i], NULL);
  g->joined[i] = 1;
}

void vx_tasks_wait(struct group *g) {
  for (int i = 0; i < g->count; i++)
    vx_task_join(g, i);
}

void vx_tasks_join(struct group *g) { vx_tasks_wait(g); }

void vx_tasks_end(struct group *g) {
  vx_tasks_wait(g);
  for (int i = 0; i < g->count; i++)
    free(g->envs[i]);
  free(g);
}

// No block is cancelled, no task fails, and the clock never moves.
int vx_cancelled(void) { return 0; }
int vx_cancelled_at(const char *at_file, int at_line) { return 0; }
void vx_task_failed(void) { assert(0 && "a task failed"); }
long vx_clock_ns(void) { return 0; }
void vx_waiting_for(int what, const char *at_file, int at_line,
                    const char *made_file, int made_line, long *channel) {}
void vx_waiting_for_lock(const char *at_file, int at_line,
                         const char *made_file, int made_line, long *holder) {}
// The holder word of a `Mutex` is only for the deadlock report.
void vx_lock_taken(long *holder) {}
void vx_lock_released(long *holder) {}
// Which task holds which channel end is only for the deadlock report too.
void vx_task_holds_an_end(long *words, int sender) {}

// The wait and the wake, as the Rust runtime does them: a lock and a condition
// variable. Here the condition variable is a count of wakes. A wait reads the
// word and the count under the lock. Then it sleeps until the count changes,
// and takes the lock again. A wake adds 1 to the count under the lock. GenMC
// turns the sleep loop into a blocked thread. With
// `--check-liveness`, it reports a thread that sleeps forever, so it also finds
// a lost wake.
static pthread_mutex_t bucket = PTHREAD_MUTEX_INITIALIZER;
static atomic_long wakes;

void vx_wait_while(atomic_long *word, long value) {
  pthread_mutex_lock(&bucket);
  if (atomic_load_explicit(word, memory_order_seq_cst) != value) {
    pthread_mutex_unlock(&bucket);
    return;
  }
  long seen = atomic_load_explicit(&wakes, memory_order_relaxed);
  pthread_mutex_unlock(&bucket);
  while (atomic_load_explicit(&wakes, memory_order_relaxed) == seen)
    ;
  pthread_mutex_lock(&bucket);
  pthread_mutex_unlock(&bucket);
}

void vx_wait_while_or_cancel(atomic_long *word, long value) {
  vx_wait_while(word, value);
}

void vx_wait_while_until(atomic_long *word, long value, long deadline,
                         int or_cancel) {
  vx_wait_while(word, value);
}

void vx_wake_all(atomic_long *word) {
  pthread_mutex_lock(&bucket);
  atomic_fetch_add_explicit(&wakes, 1, memory_order_relaxed);
  pthread_mutex_unlock(&bucket);
}

// The two words of `select`. The clients do not use `select`, but each `send`
// and `recv` touches them.
static atomic_long any_channel[2];
atomic_long *vx_any_channel_words(void) { return any_channel; }

// The skip set holds the sends that a cancel left empty. No task is cancelled
// here.
void vx_channel_skip(atomic_long *words, long k) {
  assert(0 && "no send is cancelled");
}
int vx_channel_take_skip(atomic_long *words, long k) { return 0; }
void vx_channel_forget_skips(atomic_long *words) {}

// Printing and signals. A panic prints its message and calls `abort`, and GenMC
// reports the `abort`.
long write(int fd, const void *buf, long n) { return n; }
int print_i32(int v) { return 0; }
int print_str(const char *s) { return 0; }
int vx_print_to_stderr(int on) { return 0; }
void vx_init_signals(void) {}
void vx_cancel_off(double limit) {}
void vx_cancel_on(void) {}

// GenMC runs `llvm.memset` only on a local or a global. LLVM merges the stores
// that set the words of a new channel into one `llvm.memset` on the heap. So
// run_genmc.py points the calls at this function, which writes words of 8
// bytes, as the loads of the words read them.
void vx_genmc_memset(void *dst, char value, long n, _Bool is_volatile) {
  unsigned char b = (unsigned char)value;
  if (n % 8 == 0 && (unsigned long)dst % 8 == 0) {
    long word = 0;
    for (int i = 0; i < 8; i++)
      word = (word << 8) | b;
    for (long i = 0; i < n / 8; i++)
      ((long *)dst)[i] = word;
  } else {
    for (long i = 0; i < n; i++)
      ((unsigned char *)dst)[i] = b;
  }
}
