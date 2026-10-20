# Change proposal: let the protected engine use the L4 (NVENC)

**Status: proposal only. Not implemented. `pipeline/build_video.py` is
unchanged.** It needs your explicit approval before anything is written.

## Why it would be needed

Today the L4's GPU does nothing. Every encode in the engine is `libx264` on
the CPU:

| Line | Stage | Encode |
|---|---|---|
| `build_video.py:865` | `cut` (one per segment) | `-c:v libx264 -preset $PRESET -crf $CRF -profile:v high -pix_fmt yuv420p` |
| `build_video.py:910` | normalising an input that needs it | same |
| `build_video.py:1038` | `join` (crossfades) | same |

`g2-standard-4` has 4 vCPUs. libx264 on 4 vCPUs is the slowest part of every
job, and it is exactly the work NVENC moves to dedicated hardware on the L4.
Until that happens, a CPU-optimised VM with more cores renders faster for
less money (see the cost analysis).

## The smallest change that would do it

One encoder setting, read the way `AT_CRF` and `AT_PRESET` already are, with
**today's behaviour as the default**:

```python
# near line 49, beside CRF and PRESET
ENCODER = os.environ.get("AT_ENCODER") or "libx264"     # or "h264_nvenc"

def venc():
    """The -c:v arguments for the configured encoder. Unset = unchanged."""
    if ENCODER == "h264_nvenc":
        # constant-quality VBR, the NVENC equivalent of CRF
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", CRF, "-b:v", "0",
                "-profile:v", "high", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
            "-profile:v", "high", "-pix_fmt", "yuv420p"]
```

The three call sites then replace their inline `"-c:v", "libx264", ...,
"-pix_fmt", "yuv420p"` with `*venc()`. Nothing else changes: no stage
contract, no filtergraph, no output name.

The filters (`crop`, `scale`, `gblur`, `drawtext`, `xfade`, `overlay`) stay
on the CPU. Only the encode moves. A full-GPU filter chain (`scale_cuda`,
`overlay_cuda`) would be a much larger change and is not proposed.

## What has to be true first

* ffmpeg on the worker VM is built with NVENC (`ffmpeg -encoders | grep
  nvenc`). The Ubuntu `apt` build usually is; this was **not verified**,
  because the VM could not be inspected.
* NVIDIA drivers are installed (`nvidia-smi` works). Then set
  `AT_CLOUD_REQUIRE_GPU=1` so a VM without a working GPU refuses work instead
  of silently rendering on the CPU.

## What would have to be checked

* **Quality**: NVENC at `-cq 18` is not bit-identical to x264 at `-crf 18`.
  Render the same job both ways and compare with the critic's own checks,
  plus a visual pass, before switching the default for the cloud worker.
* **The critic's technical checks** (frame count, duration, fps, codec,
  loudness) must still pass. NVENC writes the same `h264` + `yuv420p` they
  check for.
* **Determinism**: none of the artifact caching depends on encoded bytes, so
  this does not invalidate caches. It does change the MP4's md5, which only
  matters to the upload-skip check on a retry, and that compares the file
  against itself.

## If approved

1. Add `ENCODER` / `venc()` and replace the three call sites (about 15
   lines).
2. A test: with `AT_ENCODER` unset, every command `build_video` builds is
   byte-identical to today's.
3. A test with `AT_ENCODER=h264_nvenc` checking the arguments built
   (no GPU needed to check arguments).
4. One real render on the L4, compared with a CPU render of the same job.
