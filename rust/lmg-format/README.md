# lmg-format

Reads and writes **LMG Lean**, the compact on-disk form of a trained LMG scene.

An LMG model is a triangle mesh plus Gaussian splats anchored to its faces. Most of what a splat needs — where it sits, how big it is, how it is oriented — follows from the triangle it is anchored to, so Lean does not store any of it. This crate puts it back.

## Full and Lean

Training produces **Full**. It is what the training and rendering scripts read and write, and it does not change.

```
<mesh>.ply                                 lives in the mesh library, NOT in the run directory
output/<exp>/<scene>/<cfg>/point_cloud/iteration_<N>/
  point_cloud.ply                          every splat, 62 floats each
  model_params.pt                          a Python pickle (torch.save)
```

Full is convenient for training and bad for shipping. `point_cloud.ply` stores positions, scales and rotations that are all recomputed from the mesh on load anyway — plus a `normals` field that is entirely zeros. `model_params.pt` stores one budget number for *every* face in the mesh, when around 1% of faces carry a splat. And being a pickle, reading it outside Python means embedding Python.

**Lean** is the shipping form. Same information, nothing derivable, no pickle:

```
bicycle_distortion_r4.lmg/
  mesh.ply             triangle mesh, ordinary binary PLY
  scene.safetensors    appearance + the few parameters the mesh cannot imply
```

Measured on real checkpoints, Lean is 1.9x to 2.3x smaller than Full and converts back bit-for-bit. What it drops:

| dropped | why |
| --- | --- |
| `xyz` `[N,3]` | barycentric coordinates on the anchor triangle |
| `normals` `[N,3]` | always zero |
| `scaling` `[N,3]` | from the triangle's edge lengths |
| `rotation` `[N,4]` | from the triangle's normal and edges; all splats on a face share it |
| `triangle_indices` `[N]` | implied by the budget array |
| `alpha_indices` `[N]` | implied by the budget array |
| `_alpha` `[N,3]` | when splats draw from a shared position pool |
| `_scale` `[N,1]` | when all splat sizes are 1.0 |
| dense budgets `[F]` | stored as pairs for occupied faces only |

The last three are conditional: the encoder checks each one and falls back to storing the real array if a checkpoint disagrees. The file is lossless either way, and the metadata says which path was taken.

## Where this sits

```
train -> Full -> deflate -> Lean -> streaming server
                              |
                              +-> this crate: derive() -> renderable splats
```

Two independent implementations, on purpose:

* **Python** (`lmg_format/` in the repo root) converts between Full and Lean. It is the only side that touches pickle.
* **Rust** (this crate) reads and writes Lean and rebuilds the derived geometry. It never sees a `.pt`.

They do not call each other. That is what makes the conformance test meaningful — it checks one implementation against the other, not against itself.

## The file

`scene.safetensors` holds named arrays. `N` = splats, `F` = mesh faces, `K` = faces carrying at least one splat, `P` = size of the shared position pool.

| array | shape | type | present |
| --- | --- | --- | --- |
| `f_dc` | `[N,3]` | f32 | always — base colour |
| `f_rest` | `[N,45]` | f32 | always — higher spherical-harmonic bands, ~90% of the file |
| `opacity` | `[N,1]` | f32 | always |
| `budget_face_id` | `[K]` | u32 | always — which faces carry splats |
| `budget_count` | `[K]` | u8/u16 | always — how many each carries |
| `alpha_pool` | `[P,3]` | f32 | `barycentric_mode == "pool"` |
| `alpha` | `[N,3]` | f32 | `barycentric_mode == "raw"` |
| `scale` | `[N,1]` | f32 | `scale_mode == "raw"` |
| `hover` | `[N,1]` | f32 | `model_variant == "hover"` |
| `round_id` | `[N]` | u8 | if trained progressively |

Metadata (safetensors requires string values):

| key | meaning |
| --- | --- |
| `format_version` | bump when the derivation changes |
| `model_variant` | `binding` (splats on the surface) or `hover` (splats offset along the normal) |
| `sh_degree` | 3 in every current model |
| `num_splats`, `num_faces` | `N` and `F` |
| `barycentric_mode` | `pool` or `raw` |
| `scale_mode` | `unit` or `raw` |
| `mesh_sha256` | of `mesh.ply` as written; the decoder checks it |
| `mesh_normalized` | whether the mesh was rewritten as the model sees it |
| `mesh_file` | `mesh.ply` |
| `source` | where it came from, for provenance |

Arrays are stored one field at a time, not one splat at a time. So any single field is a contiguous byte range the header points at — you can fetch just `f_dc` without reading `f_rest`, which is what makes a coarse-then-refine transfer possible later. Nothing here implements that.

Two notes on the mesh. `mesh.ply` is written as the model actually sees it, not copied verbatim from the mesh library: the Python mesh loader merges near-duplicate vertices, and storing the pre-merge file would make this crate derive slightly different triangles. And the mesh is never recorded anywhere in a run directory, so the conversion step is told which mesh to use and stamps its hash here — that hash is the only thing standing between you and rendering a model against the wrong mesh.

## Using it

```toml
[dependencies]
lmg-format = { path = "../LMG_Codebase/rust/lmg-format" }
```

```rust
use lmg_format::Bundle;

let bundle = Bundle::open(std::path::Path::new("bicycle_distortion_r4.lmg"))?;
let d = bundle.derive()?;

// rebuilt geometry
// d.xyz              [N] splat centres
// d.scaling          [N] log-space scales, as the rasterizer wants them
// d.rotation         [N] quaternions, real part first
// d.triangle_indices [N] which face each splat belongs to
// d.alpha_indices    [N] which pool slot each splat used

// appearance, straight off the bundle
// bundle.f_dc, bundle.f_rest, bundle.opacity
// bundle.round_id    which training round added each splat, if present
```

`derive()` reads `mesh.ply` from the bundle and verifies its hash. If you already hold the mesh, `derive_with_mesh(&mesh)` skips the read. `write_params(&bundle, path)` writes the params file back out, preserving metadata keys this crate does not model.

Errors are `String`. Nothing panics on a malformed file; a mesh whose face count disagrees with the metadata, a missing array, an unknown mode and a future `format_version` all come back as errors.

```
cargo run --release --bin lmg -- inspect path/to/scene.lmg
```

## Sample bundles

A set of bundles converted from real trained scenes is kept outside the repo — they run from tens of megabytes to a few hundred, mostly mesh. Ask whoever handed you this crate for the directory.

| bundle | size | variant | modes | what it exercises |
| --- | --- | --- | --- | --- |
| `hotdog_binding_32k` | 25M | binding | pool / unit | the common case — start here |
| `hotdog_rawalpha_32k` | 25M | binding | raw / unit | the raw-barycentric fallback |
| `hotdog_hover_80k` | 34M | hover | pool / unit | the `hover` and `round_id` arrays |
| `ship_hover_80k` | 42M | hover | pool / unit | a different mesh |
| `bicycle_hover_320k` | 224M | hover | pool / unit | 320k splats over an 8.8M-face mesh |

Between them they cover both model variants, both barycentric modes, and the optional arrays. `scale_mode: "raw"` is not represented because no trained checkpoint has produced a non-unit scale — that path is only reachable from a synthetic fixture.

You cannot produce a bundle from Rust. Converting a trained model to Lean is the Python side's job, so new scenes have to come from whoever runs the training.

## What it deliberately does not do

It cannot read Full (`model_params.pt`) — that is the Python side's job, and avoiding pickle is the point.

It does no tiling, chunking, ordering or transport. That is the streaming server's business; this crate hands you splats.

Splats come out in triangle order, not importance order.

## Tests

```
cargo test
```

Conformance vectors under `tests/vectors/` are generated by the Python side through the real model code, covering both `binding` and `hover` and both fallback modes. To also run against a real scene (too large to commit):

```
python -m lmg_format.make_vectors        # regenerates the committed vectors
LMG_REAL_VECTOR=/tmp/lmg_real_vector cargo test --release -- --nocapture
```

Tolerances are not bit-exact and cannot be. torch accumulates sums in its own order; any other implementation differs in the last bit. Two spots magnify that: scales are stored as a log, so on thin triangles one ULP becomes ~1e-4 in log space; and quaternions come from a Gram-Schmidt frame that is ill-conditioned on near-degenerate faces. Measured against a real 32k-splat scene: positions agree to 3.6e-7, scales (compared as actual scales, not logs) to 2.5e-7, rotations to 7.1e-5 worst case with a median of 6.0e-7 — about 0.008 degrees. Each field is checked on both its worst case and its median; the median is what catches a genuine mistake, since a real bug shifts the whole distribution rather than just the tail.

## Layout

| file | what |
| --- | --- |
| `src/lib.rs` | bundle reading and writing, metadata |
| `src/derive.rs` | the rebuild — positions, scales, rotations |
| `src/ply.rs` | minimal binary PLY mesh reader |
| `src/bin/lmg.rs` | `inspect` |

`derive.rs` mirrors `prepare_scaling_rot`, `update_alpha` and `_calc_xyz` in `games/mesh_splatting/scene/gaussian_mesh_model.py`, and `rot_to_quat_batch` in `utils/general_utils.py`. If those change, this changes with them and `format_version` goes up.
