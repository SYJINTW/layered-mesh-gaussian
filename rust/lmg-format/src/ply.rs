//! Minimal binary-little-endian PLY mesh reader.
//!
//! ponytail: hand-rolled instead of pulling in a PLY crate. The only PLY this
//! crate ever reads is the mesh emitted by trimesh: binary_little_endian, a
//! `vertex` element with float x/y/z (plus colour properties we skip) and a
//! `face` element with one `list uchar int` property. Ceiling: ascii PLY and
//! big-endian are rejected, not parsed. Swap in a real PLY crate if the mesh
//! source ever changes.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

fn scalar_size(ty: &str) -> Option<usize> {
    Some(match ty {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "int32" | "uint32" | "float" | "float32" => 4,
        "double" | "float64" | "int64" | "uint64" => 8,
        _ => return None,
    })
}

enum Prop {
    Scalar { name: String, ty: String },
    List { count_ty: String, item_ty: String },
}

struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

fn read_u32(buf: &[u8], off: usize, ty: &str) -> u32 {
    match ty {
        "uchar" | "uint8" | "char" | "int8" => buf[off] as u32,
        "short" | "ushort" | "int16" | "uint16" => {
            u16::from_le_bytes([buf[off], buf[off + 1]]) as u32
        }
        _ => u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]),
    }
}

pub fn read_mesh(path: &Path) -> Result<Mesh, String> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let mut raw = Vec::new();
    reader.read_to_end(&mut raw).map_err(|e| e.to_string())?;

    // Header is ascii and ends at "end_header\n".
    let marker = b"end_header\n";
    let hdr_end = raw
        .windows(marker.len())
        .position(|w| w == marker)
        .ok_or("no end_header")?
        + marker.len();
    let header = std::str::from_utf8(&raw[..hdr_end]).map_err(|e| e.to_string())?;

    if !header.contains("format binary_little_endian") {
        return Err("only binary_little_endian PLY is supported".into());
    }

    let mut elements: Vec<Element> = Vec::new();
    for line in header.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["element", name, count] => elements.push(Element {
                name: (*name).to_string(),
                count: count.parse().map_err(|_| "bad element count")?,
                props: Vec::new(),
            }),
            ["property", "list", count_ty, item_ty, _] => {
                let e = elements.last_mut().ok_or("property before element")?;
                e.props.push(Prop::List {
                    count_ty: (*count_ty).to_string(),
                    item_ty: (*item_ty).to_string(),
                });
            }
            ["property", ty, name] => {
                let e = elements.last_mut().ok_or("property before element")?;
                e.props.push(Prop::Scalar {
                    name: (*name).to_string(),
                    ty: (*ty).to_string(),
                });
            }
            _ => {}
        }
    }

    let body = &raw[hdr_end..];
    let mut off = 0usize;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    for el in &elements {
        let is_vertex = el.name == "vertex";
        let is_face = el.name == "face";
        // Reserve against what the file can actually contain, not against a
        // count the header claims -- a corrupt count asked for tens of GB.
        let cap = el.count.min(body.len().saturating_sub(off) / 4 + 1);
        if is_vertex {
            vertices.reserve(cap);
        }
        if is_face {
            faces.reserve(cap);
        }

        for _ in 0..el.count {
            let mut xyz = [0f32; 3];
            for prop in &el.props {
                match prop {
                    Prop::Scalar { name, ty } => {
                        let sz = scalar_size(ty).ok_or_else(|| format!("unknown type {}", ty))?;
                        if off + sz > body.len() {
                            return Err("truncated PLY body".into());
                        }
                        if is_vertex {
                            let slot = match name.as_str() {
                                "x" => Some(0),
                                "y" => Some(1),
                                "z" => Some(2),
                                _ => None,
                            };
                            if let Some(i) = slot {
                                if sz != 4 {
                                    return Err("vertex coords must be float32".into());
                                }
                                xyz[i] = f32::from_le_bytes([
                                    body[off],
                                    body[off + 1],
                                    body[off + 2],
                                    body[off + 3],
                                ]);
                            }
                        }
                        off += sz;
                    }
                    Prop::List { count_ty, item_ty } => {
                        let csz = scalar_size(count_ty).ok_or("bad list count type")?;
                        if off + csz > body.len() {
                            return Err("truncated PLY body".into());
                        }
                        let n = read_u32(body, off, count_ty) as usize;
                        off += csz;
                        let isz = scalar_size(item_ty).ok_or("bad list item type")?;
                        if off + isz * n > body.len() {
                            return Err("truncated PLY body".into());
                        }
                        if is_face {
                            if n != 3 {
                                return Err("mesh must be triangulated".into());
                            }
                            faces.push([
                                read_u32(body, off, item_ty),
                                read_u32(body, off + isz, item_ty),
                                read_u32(body, off + 2 * isz, item_ty),
                            ]);
                        }
                        off += isz * n;
                    }
                }
            }
            if is_vertex {
                vertices.push(xyz);
            }
        }
    }
    // Face indices are used to slice `vertices` directly, so an out-of-range
    // one is a panic rather than an error unless it is caught here.
    let nv = vertices.len() as u32;
    if let Some(bad) = faces.iter().flatten().find(|i| **i >= nv) {
        return Err(format!(
            "face references vertex {} but the mesh has {} vertices",
            bad, nv
        ));
    }
    Ok(Mesh { vertices, faces })
}
