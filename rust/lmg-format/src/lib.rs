//! Reader/writer for the LMG Lean scene format.
//!
//! A Lean bundle is a directory:
//!
//! ```text
//! scene.lmg/
//!   mesh.ply            standard triangle mesh
//!   scene.safetensors   appearance + the parameters the mesh cannot imply
//! ```
//!
//! Everything derivable from the mesh (splat centres, scales, rotations, the
//! splat->triangle mapping) is absent from the file and rebuilt here. This
//! crate deliberately cannot read LMG Full's `model_params.pt`: that is a
//! Python pickle, and not depending on it is the point of the format.

pub mod derive;
pub mod ply;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: &str = "1";
pub const PARAMS_NAME: &str = "scene.safetensors";

#[derive(Debug, Clone, PartialEq)]
pub enum Variant {
    Binding,
    Hover,
}

#[derive(Debug, Clone)]
pub struct Meta {
    pub format_version: String,
    pub variant: Variant,
    pub sh_degree: usize,
    pub num_splats: usize,
    pub num_faces: usize,
    pub barycentric_mode: String,
    pub scale_mode: String,
    pub mesh_sha256: String,
    pub mesh_file: String,
}

/// A Lean bundle with its arrays decoded but nothing derived yet.
pub struct Bundle {
    pub meta: Meta,
    /// Every metadata key as stored, including ones this crate does not model.
    /// `write_params` emits this verbatim so a rewrite is never lossy.
    pub metadata: HashMap<String, String>,
    pub dir: PathBuf,
    pub f_dc: Vec<f32>,
    pub f_rest: Vec<f32>,
    pub opacity: Vec<f32>,
    pub budgets: Vec<u32>,
    pub alpha_pool: Option<Vec<[f32; 3]>>,
    pub alpha_raw: Option<Vec<[f32; 3]>>,
    pub scale_raw: Option<Vec<f32>>,
    pub hover: Option<Vec<f32>>,
    pub round_id: Option<Vec<u8>>,
}

/// Everything LMG Lean leaves out, rebuilt.
pub struct Derived {
    pub triangle_indices: Vec<u32>,
    pub alpha_indices: Vec<u32>,
    pub xyz: Vec<[f32; 3]>,
    pub scaling: Vec<[f32; 3]>,
    pub rotation: Vec<[f32; 4]>,
}

fn f32s(t: &TensorView) -> Result<Vec<f32>, String> {
    if t.dtype() != Dtype::F32 {
        return Err(format!("expected f32, got {:?}", t.dtype()));
    }
    Ok(t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn triples(v: Vec<f32>) -> Vec<[f32; 3]> {
    v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn uints(t: &TensorView) -> Vec<u32> {
    match t.dtype() {
        Dtype::U8 => t.data().iter().map(|b| *b as u32).collect(),
        Dtype::U16 => t
            .data()
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
            .collect(),
        _ => t
            .data()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

fn need<'a>(m: &'a HashMap<String, String>, k: &str) -> Result<&'a String, String> {
    m.get(k).ok_or_else(|| format!("missing metadata key {}", k))
}

impl Bundle {
    pub fn open(dir: &Path) -> Result<Bundle, String> {
        let params = dir.join(PARAMS_NAME);
        let raw = std::fs::read(&params).map_err(|e| format!("{}: {}", params.display(), e))?;
        let st = SafeTensors::deserialize(&raw).map_err(|e| e.to_string())?;
        let (_, header) = SafeTensors::read_metadata(&raw).map_err(|e| e.to_string())?;
        let md = header
            .metadata()
            .clone()
            .ok_or("bundle has no __metadata__")?;

        let meta = Meta {
            format_version: need(&md, "format_version")?.clone(),
            variant: match need(&md, "model_variant")?.as_str() {
                "hover" => Variant::Hover,
                "binding" => Variant::Binding,
                other => return Err(format!("unknown model_variant {}", other)),
            },
            sh_degree: need(&md, "sh_degree")?.parse().map_err(|_| "bad sh_degree")?,
            num_splats: need(&md, "num_splats")?
                .parse()
                .map_err(|_| "bad num_splats")?,
            num_faces: need(&md, "num_faces")?.parse().map_err(|_| "bad num_faces")?,
            barycentric_mode: need(&md, "barycentric_mode")?.clone(),
            scale_mode: need(&md, "scale_mode")?.clone(),
            mesh_sha256: need(&md, "mesh_sha256")?.clone(),
            mesh_file: need(&md, "mesh_file")?.clone(),
        };
        if meta.format_version != FORMAT_VERSION {
            return Err(format!(
                "bundle is format v{}, this crate reads v{}",
                meta.format_version, FORMAT_VERSION
            ));
        }

        let get = |n: &str| st.tensor(n).map_err(|e| format!("{}: {}", n, e));
        let opt = |n: &str| st.tensor(n).ok();

        let budgets = derive::densify(
            &uints(&get("budget_face_id")?),
            &uints(&get("budget_count")?),
            meta.num_faces,
        );

        Ok(Bundle {
            f_dc: f32s(&get("f_dc")?)?,
            f_rest: f32s(&get("f_rest")?)?,
            opacity: f32s(&get("opacity")?)?,
            budgets,
            alpha_pool: match opt("alpha_pool") {
                Some(t) => Some(triples(f32s(&t)?)),
                None => None,
            },
            alpha_raw: match opt("alpha") {
                Some(t) => Some(triples(f32s(&t)?)),
                None => None,
            },
            scale_raw: match opt("scale") {
                Some(t) => Some(f32s(&t)?),
                None => None,
            },
            hover: match opt("hover") {
                Some(t) => Some(f32s(&t)?),
                None => None,
            },
            round_id: opt("round_id").map(|t| t.data().to_vec()),
            meta,
            metadata: md,
            dir: dir.to_path_buf(),
        })
    }

    /// Rebuild splat centres, scales and rotations from the mesh.
    pub fn derive(&self) -> Result<Derived, String> {
        let mesh = ply::read_mesh(&self.dir.join(&self.meta.mesh_file))?;
        self.derive_with_mesh(&mesh)
    }

    pub fn derive_with_mesh(&self, mesh: &ply::Mesh) -> Result<Derived, String> {
        if mesh.faces.len() != self.meta.num_faces {
            return Err(format!(
                "mesh has {} faces, bundle declares {}",
                mesh.faces.len(),
                self.meta.num_faces
            ));
        }
        let triangles: Vec<[[f32; 3]; 3]> = mesh
            .faces
            .iter()
            .map(|f| {
                [
                    mesh.vertices[f[0] as usize],
                    mesh.vertices[f[1] as usize],
                    mesh.vertices[f[2] as usize],
                ]
            })
            .collect();

        let (triangle_indices, alpha_indices) = derive::indices(&self.budgets);
        if triangle_indices.len() != self.meta.num_splats {
            return Err(format!(
                "budgets sum to {}, bundle declares {} splats",
                triangle_indices.len(),
                self.meta.num_splats
            ));
        }

        let raw_alpha: Vec<[f32; 3]> = match self.meta.barycentric_mode.as_str() {
            "pool" => {
                let pool = self.alpha_pool.as_ref().ok_or("pool mode without alpha_pool")?;
                alpha_indices.iter().map(|i| pool[*i as usize]).collect()
            }
            "raw" => self.alpha_raw.clone().ok_or("raw mode without alpha")?,
            other => return Err(format!("unknown barycentric_mode {}", other)),
        };
        let alpha = derive::normalize_alpha(&raw_alpha);

        let scale = match self.meta.scale_mode.as_str() {
            "unit" => vec![1.0f32; self.meta.num_splats],
            "raw" => self.scale_raw.clone().ok_or("raw scale_mode without scale")?,
            other => return Err(format!("unknown scale_mode {}", other)),
        };

        let frames = derive::face_frames(&triangles);
        let mut xyz = derive::calc_xyz(&alpha, &triangle_indices, &triangles);
        if self.meta.variant == Variant::Hover {
            let hover = self.hover.as_ref().ok_or("hover variant without hover")?;
            derive::apply_hover(&mut xyz, hover, &triangle_indices, &triangles);
        }

        Ok(Derived {
            scaling: derive::calc_scaling(&scale, &triangle_indices, &frames),
            rotation: derive::calc_rotation(&triangle_indices, &frames),
            xyz,
            triangle_indices,
            alpha_indices,
        })
    }
}

/// Re-serialize a bundle's params file. Byte-identical to the input when
/// nothing was changed, which is what the round-trip test asserts.
pub fn write_params(bundle: &Bundle, out: &Path) -> Result<(), String> {
    let md = bundle.metadata.clone();

    let n = bundle.meta.num_splats;
    let mut face_id: Vec<u8> = Vec::new();
    let mut count: Vec<u8> = Vec::new();
    for (f, c) in bundle.budgets.iter().enumerate() {
        if *c != 0 {
            face_id.extend_from_slice(&(f as u32).to_le_bytes());
            count.push(*c as u8);
        }
    }
    let k = count.len();

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let f_dc = bytes(&bundle.f_dc);
    let f_rest = bytes(&bundle.f_rest);
    let opacity = bytes(&bundle.opacity);
    let n_rest = bundle.f_rest.len() / n.max(1);

    let mut views: Vec<(String, TensorView)> = vec![
        ("f_dc".into(), TensorView::new(Dtype::F32, vec![n, 3], &f_dc).map_err(|e| e.to_string())?),
        ("f_rest".into(), TensorView::new(Dtype::F32, vec![n, n_rest], &f_rest).map_err(|e| e.to_string())?),
        ("opacity".into(), TensorView::new(Dtype::F32, vec![n, 1], &opacity).map_err(|e| e.to_string())?),
        ("budget_face_id".into(), TensorView::new(Dtype::U32, vec![k], &face_id).map_err(|e| e.to_string())?),
        ("budget_count".into(), TensorView::new(Dtype::U8, vec![k], &count).map_err(|e| e.to_string())?),
    ];

    // Owned buffers must outlive the views, so build them before the vec grows.
    let pool_b = bundle.alpha_pool.as_ref().map(|p| bytes(&p.concat()));
    let alpha_b = bundle.alpha_raw.as_ref().map(|p| bytes(&p.concat()));
    let scale_b = bundle.scale_raw.as_ref().map(|s| bytes(s));
    let hover_b = bundle.hover.as_ref().map(|h| bytes(h));

    if let (Some(b), Some(p)) = (&pool_b, &bundle.alpha_pool) {
        views.push(("alpha_pool".into(), TensorView::new(Dtype::F32, vec![p.len(), 3], b).map_err(|e| e.to_string())?));
    }
    if let Some(b) = &alpha_b {
        views.push(("alpha".into(), TensorView::new(Dtype::F32, vec![n, 3], b).map_err(|e| e.to_string())?));
    }
    if let Some(b) = &scale_b {
        views.push(("scale".into(), TensorView::new(Dtype::F32, vec![n, 1], b).map_err(|e| e.to_string())?));
    }
    if let Some(b) = &hover_b {
        views.push(("hover".into(), TensorView::new(Dtype::F32, vec![n, 1], b).map_err(|e| e.to_string())?));
    }
    if let Some(r) = &bundle.round_id {
        views.push(("round_id".into(), TensorView::new(Dtype::U8, vec![n], r).map_err(|e| e.to_string())?));
    }

    safetensors::serialize_to_file(views, &Some(md), out).map_err(|e| e.to_string())
}
