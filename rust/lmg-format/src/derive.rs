//! Rebuild the fields LMG Lean drops.
//!
//! Ports `update_alpha`, `_calc_xyz` and `prepare_scaling_rot` from
//! games/mesh_splatting/scene/gaussian_mesh_model.py and `rot_to_quat_batch`
//! from utils/general_utils.py. Kept in the same operation order as the torch
//! code so the outputs match within the conformance tolerance.

pub const EPS_S0: f32 = 1e-8;

/// Sparse (face_id, count) pairs -> dense per-face budget array.
pub fn densify(face_id: &[u32], count: &[u32], num_faces: usize) -> Vec<u32> {
    let mut dense = vec![0u32; num_faces];
    for (f, c) in face_id.iter().zip(count) {
        dense[*f as usize] = *c;
    }
    dense
}

/// `triangle_indices = repeat(arange(F), budgets)` and
/// `alpha_indices = concat(arange(n) for n in budgets)`.
pub fn indices(budgets: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let total: usize = budgets.iter().map(|b| *b as usize).sum();
    let mut tri = Vec::with_capacity(total);
    let mut alpha = Vec::with_capacity(total);
    for (f, n) in budgets.iter().enumerate() {
        for k in 0..*n {
            tri.push(f as u32);
            alpha.push(k);
        }
    }
    (tri, alpha)
}

/// `relu(_alpha) + 1e-8`, normalized to sum 1 per splat.
pub fn normalize_alpha(raw: &[[f32; 3]]) -> Vec<[f32; 3]> {
    raw.iter()
        .map(|a| {
            let r = [
                a[0].max(0.0) + 1e-8,
                a[1].max(0.0) + 1e-8,
                a[2].max(0.0) + 1e-8,
            ];
            let s = r[0] + r[1] + r[2];
            [r[0] / s, r[1] / s, r[2] / s]
        })
        .collect()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn scaled(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

/// Per-face orthonormal frame (v0 = unit normal, v1, v2) and the (s0, s1, s2)
/// half-extents that `prepare_scaling_rot` multiplies `_scale` into.
pub struct FaceFrame {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub s: [f32; 3],
}

pub fn face_frames(triangles: &[[[f32; 3]; 3]]) -> Vec<FaceFrame> {
    triangles
        .iter()
        .map(|t| {
            let normals = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let v0 = scaled(normals, 1.0 / (norm(normals) + EPS_S0));

            let means = [
                (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                (t[0][2] + t[1][2] + t[2][2]) / 3.0,
            ];
            let v1_raw = sub(t[1], means);
            let v1_norm = norm(v1_raw) + EPS_S0;
            let v1 = scaled(v1_raw, 1.0 / v1_norm);

            let v2_init = sub(t[2], means);
            // Gram-Schmidt against v0 and v1.
            let p0 = scaled(v0, dot(v2_init, v0));
            let p1 = scaled(v1, dot(v2_init, v1));
            let v2_raw = sub(sub(v2_init, p0), p1);
            let v2 = scaled(v2_raw, 1.0 / (norm(v2_raw) + EPS_S0));

            let s1 = v1_norm / 2.0;
            let s2 = dot(v2_init, v2) / 2.0;
            FaceFrame {
                v0,
                v1,
                v2,
                s: [EPS_S0, s1, s2],
            }
        })
        .collect()
}

/// Splat centre: barycentric combination of the anchor triangle's vertices.
pub fn calc_xyz(
    alpha: &[[f32; 3]],
    triangle_indices: &[u32],
    triangles: &[[[f32; 3]; 3]],
) -> Vec<[f32; 3]> {
    alpha
        .iter()
        .zip(triangle_indices)
        .map(|(a, ti)| {
            let t = &triangles[*ti as usize];
            let mut out = [0f32; 3];
            for c in 0..3 {
                out[c] = a[0] * t[0][c] + a[1] * t[1][c] + a[2] * t[2][c];
            }
            out
        })
        .collect()
}

/// `LMGModelHover.hover_activation`: asymmetric scaled tanh -- full tanh on the
/// outward side, damped by `hover_eps` on the inward side.
pub const HOVER_EPS: f32 = 0.05;

pub fn hover_activation(x: f32) -> f32 {
    let t = x.tanh();
    if x >= 0.0 {
        t
    } else {
        HOVER_EPS * t
    }
}

/// Hover variant: offset each centre along the face normal.
/// Mirrors `LMGModelHover._face_normals_and_hover_scales` + `_calc_xyz`.
pub fn apply_hover(
    xyz: &mut [[f32; 3]],
    hover: &[f32],
    triangle_indices: &[u32],
    triangles: &[[[f32; 3]; 3]],
) {
    let per_face: Vec<([f32; 3], f32)> = triangles
        .iter()
        .map(|t| {
            let e12 = sub(t[1], t[0]);
            let e13 = sub(t[2], t[0]);
            let n = cross(e12, e13);
            let nn = norm(n);
            let unit = scaled(n, 1.0 / (nn + EPS_S0));
            (unit, norm(e12) * norm(e13) / (nn + EPS_S0))
        })
        .collect();
    for (i, p) in xyz.iter_mut().enumerate() {
        let (unit, hs) = per_face[triangle_indices[i] as usize];
        let k = hover_activation(hover[i]) * hs;
        for c in 0..3 {
            p[c] += unit[c] * k;
        }
    }
}

/// `log(relu(_scale * [eps, s1, s2]) + eps)`, the log-space scale the PLY stores.
pub fn calc_scaling(
    scale: &[f32],
    triangle_indices: &[u32],
    frames: &[FaceFrame],
) -> Vec<[f32; 3]> {
    triangle_indices
        .iter()
        .enumerate()
        .map(|(i, ti)| {
            let s = frames[*ti as usize].s;
            let mut out = [0f32; 3];
            for c in 0..3 {
                out[c] = ((scale[i] * s[c]).max(0.0) + EPS_S0).ln();
            }
            out
        })
        .collect()
}

fn sqrt_positive(x: f32) -> f32 {
    if x > 0.0 {
        x.sqrt()
    } else {
        0.0
    }
}

/// pytorch3d's `matrix_to_quaternion`, real part first, standardized.
fn rot_to_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let (m00, m01, m02) = (m[0][0], m[0][1], m[0][2]);
    let (m10, m11, m12) = (m[1][0], m[1][1], m[1][2]);
    let (m20, m21, m22) = (m[2][0], m[2][1], m[2][2]);

    let q_abs = [
        sqrt_positive(1.0 + m00 + m11 + m22),
        sqrt_positive(1.0 + m00 - m11 - m22),
        sqrt_positive(1.0 - m00 + m11 - m22),
        sqrt_positive(1.0 - m00 - m11 + m22),
    ];
    let candidates = [
        [q_abs[0] * q_abs[0], m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, q_abs[1] * q_abs[1], m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, q_abs[2] * q_abs[2], m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, q_abs[3] * q_abs[3]],
    ];

    // Pick the best-conditioned candidate (largest |q| component).
    let mut best = 0usize;
    for i in 1..4 {
        if q_abs[i] > q_abs[best] {
            best = i;
        }
    }
    let denom = 2.0 * q_abs[best].max(0.1);
    let mut q = [
        candidates[best][0] / denom,
        candidates[best][1] / denom,
        candidates[best][2] / denom,
        candidates[best][3] / denom,
    ];
    if q[0] < 0.0 {
        for c in q.iter_mut() {
            *c = -*c;
        }
    }
    q
}

/// Per-splat rotation quaternion. Every splat on a face shares its face's frame.
pub fn calc_rotation(triangle_indices: &[u32], frames: &[FaceFrame]) -> Vec<[f32; 4]> {
    // prepare_scaling_rot stacks (v0, v1, v2) as ROWS then transposes, so the
    // matrix handed to rot_to_quat_batch has them as COLUMNS.
    let per_face: Vec<[f32; 4]> = frames
        .iter()
        .map(|f| {
            rot_to_quat([
                [f.v0[0], f.v1[0], f.v2[0]],
                [f.v0[1], f.v1[1], f.v2[1]],
                [f.v0[2], f.v1[2], f.v2[2]],
            ])
        })
        .collect();
    triangle_indices
        .iter()
        .map(|ti| per_face[*ti as usize])
        .collect()
}
