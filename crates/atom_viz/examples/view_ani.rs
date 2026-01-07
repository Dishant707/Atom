use std::env;
use atom_core::System;
use atom_physics::ani::AniEnsemble;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -p atom_viz --example view_ani <path/to/protein.pdb>");
        eprintln!("Example: cargo run -p atom_viz --example view_ani 1crn.pdb");
        return;
    }
    
    let path = &args[1];
    println!("🧬 Loading File: {}", path);
    
    let result = if path.ends_with(".xyz") {
        System::from_xyz(path)
    } else {
        System::from_pdb(path)
    };

    match result {
        Ok(mut system) => {
            println!("✅ Successfully loaded system with {} atoms.", system.positions.len());
            
            // Inject Thermal Energy (300 Kelvin) for realistic dynamics
            println!("🔥 Injecting Thermal Energy (300K)...");
            system.initialize_maxwell_boltzmann(300.0);

            // Initialize ANI-2x Neural Potential
            let models_dir = "/Users/dishant/Downloads/ani2xr_onnx_models";
            println!("🧠 Initializing ANI-2x Neural Potential...");
            
            let force_field = match AniEnsemble::new(models_dir) {
                Ok(ani) => {
                    println!("✅ ANI-2x Brain Loaded!");
                    println!("   Models: H, C, N, O, S, F, Cl");
                    println!("   Mode: Quantum-level MD simulation");
                    Some(Box::new(ani) as Box<dyn atom_physics::ForceField>)
                },
                Err(e) => {
                    eprintln!("⚠️  Warning: Could not load ANI models: {}", e);
                    eprintln!("   Visualization will be static.");
                    None
                }
            };

            println!("\n🚀 Starting visualization...");
            println!("   Press SPACE to pause/resume");
            println!("   Use mouse to rotate/zoom");
            
            pollster::block_on(atom_viz::run(system, force_field, std::collections::HashMap::new()));
        },
        Err(e) => {
            eprintln!("❌ Error loading file: {}", e);
        }
    }
}
