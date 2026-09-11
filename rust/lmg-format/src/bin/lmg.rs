//! `lmg inspect <bundle>` -- print a Lean bundle's header and derived counts.

use lmg_format::{Bundle, Variant};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] != "inspect" {
        eprintln!("usage: lmg inspect <bundle.lmg>");
        std::process::exit(2);
    }
    let bundle = match Bundle::open(Path::new(&args[2])) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    let m = &bundle.meta;
    println!("format_version   {}", m.format_version);
    println!(
        "model_variant    {}",
        if m.variant == Variant::Hover { "hover" } else { "binding" }
    );
    println!("sh_degree        {}", m.sh_degree);
    println!("num_splats       {}", m.num_splats);
    println!("num_faces        {}", m.num_faces);
    println!("barycentric_mode {}", m.barycentric_mode);
    println!("scale_mode       {}", m.scale_mode);
    println!("mesh_file        {}", m.mesh_file);
    println!("mesh_sha256      {}", m.mesh_sha256);
    println!(
        "occupied_faces   {}",
        bundle.budgets.iter().filter(|b| **b > 0).count()
    );
    match bundle.derive() {
        Ok(d) => println!("derived          {} splats", d.xyz.len()),
        Err(e) => println!("derived          unavailable ({})", e),
    }
}
