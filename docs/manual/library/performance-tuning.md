# Performance Tuning Guide

## Managed memory

SCALE doesn't yet support the `__managed__` keyword, but managed memory may still be
used via `cudaMallocManaged` and related APIs.

If you're using managed memory, you probably want to:
- Set the `SCALE_AMD_XNACK=1` environment variable.
- Make proper use of `cudaMemPrefetchAsync` and `cudaMemAdvise`. These have a more
  dramatic effect on performance on AMD than they do for NVIDIA (but are beneficial
  on NVIDIA, too!).

SCALE's implementation of managed memory is typically much faster than HIP's.

Enabling XNACK does not make every application faster: try it out and see.

A more thorough explanation is in the next section.

### XNACK on AMD GPUs

The underlying hardware feature for page relocation works quite differently on NVIDIA
vs. AMD GPUs. In NVIDIA's implementation, merely allocating a managed memory buffer is
enough to cause it to exhibit the page relocation feature most users expect. This can
then be fine-tuned with APIs like `cudaMemAdvise`.

In AMD's implementation, a hardware feature called XNACK needs to be enabled per-process for
any relocation of pages to occur. If XNACK is disabled, a managed memory buffer will
never be relocated. When a buffer is read from the "wrong side", its pages will always
be fetched over the PCIe bus and will not be moved to the GPU to make subsequent accesses
faster. This is useful for managed memory buffers which are read only rarely (eg. a huge
lookup table where only a small fraction of the entries are ever used), but will be very
slow for other access patterns.

With XNACK disabled, managed memory is very similar to a pinned host buffer.

Enabling XNACK has roughly these effects:
- Pages will now be relocated when read from the "wrong side", as users typically expect
  from managed memory.
- Access to managed memory buffers from the "right side" will be slower.

XNACK has the following properties:
- Your BIOS settings must be exactly correct or it won't work at all (SVM and HMM enabled).
- *Every* AMD GPU in your system must support XNACK. Merely having a non-XNACK GPU connected
  is enough to make the kernel driver refuse to do it.
- It does not work on every GPU. Most gfx9 devices are supported (including the MI-series
  datacenter cards). It does not work at all on any gfx10/11 cards.
- You can mix XNACK and non-XNACK processes on the same GPU, but you cannot mix XNACK and
  non-XNACK behaviour within one process.

For even more information about XNACK, we found the following sources useful:

- [Oak Ridge Lab page on XNACK](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#enabling-gpu-page-migration)
- [niconiconi's blog post about it](https://niconiconi.neocities.org/tech-notes/xnack-on-amd-gpus).

Notes:
- SCALE does not look for the `HSA_XNACK` environment variable.
- SCALE does not require any compiler flags to enable XNACK: it's purely a runtime setting
  driven by the environment variable.
