# Profiling with scaleprof

`scaleprof` is SCALE's profiler. It records what your program does on the host and
the GPU, summarises it, reads hardware counters, shows the GPU code the compiler
produced, and writes traces you can open in [Perfetto](https://ui.perfetto.dev).
It's installed with SCALE and works with any program that uses SCALE's runtime.

A typical session:

```sh
scaleprof run ./app                               # record, and list the slowest kernels
scaleprof run --counters basic ./app              # the same, with hardware counters
scaleprof summary --brief scale_profile.scaleprof # a few lines per kernel
scaleprof isa -p scale_profile.scaleprof ./app    # the GPU code of the kernels that ran
```

Each command has more options than shown here: `scaleprof help <command>` lists them.

## Recording a profile

`scaleprof run [<options>] <program> [<args>...]` runs your program, records it and
every process it starts into `scale_profile.scaleprof`, and prints the kernels that
took the most GPU time. Useful options:

| Option                         | Description                                            |
|--------------------------------|--------------------------------------------------------|
| `-o <profile>`                 | Where to write the profile.                            |
| `--view`                       | Also write a trace and open it in Perfetto.            |
| `--from-api`                   | Record only between `cudaProfilerStart` and `Stop`.    |
| `--nvtx-range <message>`       | Record only during NVTX ranges with this message.      |
| `--counters <counters>`        | Read hardware counters around each kernel.             |
| `--kernel-name <regex>`        | Read counters only around kernels matching the regex.  |

A profile holds API calls, kernels and memory operations with their timings, the
links between them (which call launched which kernel, which event a stream waited
for), NVTX ranges, memory allocations, and details of the host and GPUs.

For MPI programs, give each rank its own profile and combine them later:

```sh
mpirun -n 4 sh -c 'scaleprof run -o out/rank$OMPI_COMM_WORLD_RANK.scaleprof ./app'
scaleprof trace -o all.json out/rank*.scaleprof
```

## Hardware counters

`--counters` takes counter names, expressions of them, or the sets `basic`,
`compute`, `memory`, `occupancy` and `stalls`. `scaleprof counters` lists what your
GPU supports.

Counters are read for the whole GPU, so `scaleprof` runs each counted kernel on its
own, and fixes the GPU's clocks while your program runs so that runs can be compared.
Counted kernels therefore run slower than usual. Work from other programs on the
same GPU is counted too.

If you ask for more counters than the GPU can read at once, `scaleprof` skips the
extra ones and names them; collect those in another run.

## Reading profiles

| Command         | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `summary`       | Statistics about kernels, API calls, copies and allocations.   |
| `diff`          | Compare the kernels of two profiles.                           |
| `trace`         | Convert profiles to a trace for Perfetto.                      |
| `query`         | Run SQL on profiles.                                           |
| `bw`            | Measure how fast your GPU streams memory.                      |

`summary` shows, for each kernel, its time, launch sizes, register and shared memory
use, and its theoretical occupancy and what limits it. When registers limit
occupancy, it says how many registers would fit more threads. With hardware counters,
each kernel also gets a likely bound (memory bandwidth, compute, the memory pipeline,
or latency) and a next step to try. These are a starting point for investigation, so
check the figures they're based on, which `summary` prints alongside.

The memory bound compares your kernel with what your GPU actually streams. Run
`scaleprof bw` once on an idle GPU to measure that; otherwise the GPU's published
peak is used.

## Traces

`scaleprof trace -o <trace> <profile>...` writes a timeline of API calls and NVTX
ranges on each host thread, and kernels and copies on each GPU stream, with arrows
from each call to the work it caused and between streams that waited for each other.
`scaleprof view <trace>` opens it in Perfetto in your browser. Over SSH, forward port
9001 first (`ssh -L 9001:127.0.0.1:9001 <host>`). Perfetto doesn't upload traces.

## Looking at the GPU code

`scaleprof isa <file>` shows the kernels in a program, library or object built with
SCALE's `nvcc`. It needs no profile and no GPU.

```sh
scaleprof isa ./app                     # registers, memory and instruction counts per kernel
scaleprof isa -d -S -k 'axpy' ./app     # the code of matching kernels, with source lines
scaleprof isa --diff before.so after.so # what changed between two builds
```

For each kernel it shows how many instructions of each kind the code contains (these
are counts of the code, not of what ran), and points out likely register spills and
vector loads that could be scalar loads. Build with `-lineinfo` to see source lines.

## Querying profiles with SQL

`scaleprof query -q '<sql>' <profile>...` loads profiles into Perfetto's
[trace processor](https://perfetto.dev/docs/analysis/trace-processor) and runs your
SQL. The `scale_*` tables cover kernels, API calls, copies, counters, NVTX ranges and
GPU idle time; their columns are described in `share/scaleprof/sql/` in your SCALE
install. For example:

```sql
-- One line per kernel, most GPU time first.
SELECT * FROM scale_kernel_brief;

-- Was the GPU waiting for the host?
SELECT * FROM scale_gpu_utilisation;
```
