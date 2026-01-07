use atom_core::{System, SpatialGrid};
use atom_physics::ani::AniEnsemble;
use atom_physics::ForceField;
use glam::Vec3;

fn main() {
    // 1. Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Check for --fetch flag
    let (fetch_mode, pdb_id) = if args.len() > 2 && args[1] == "--fetch" {
        (true, Some(args[2].clone()))
    } else {
        (false, None)
    };
    
    let xyz_path = if fetch_mode {
        // Download from PDB
        let id = pdb_id.as_ref().unwrap();
        let url = format!("https://files.rcsb.org/download/{}.cif", id.to_uppercase());
        let filename = format!("{}.cif", id.to_uppercase());
        
        println!("🌐 Fetching {} from RCSB PDB...", id.to_uppercase());
        
        // Use curl to download
        let output = std::process::Command::new("curl")
            .args(&["-s", "-o", &filename, &url])
            .output()
            .expect("Failed to execute curl");
        
        if !output.status.success() {
            eprintln!("❌ Failed to download {}. Check your internet connection or PDB ID.", id);
            std::process::exit(1);
        }
        
        // Check if file was created and has content
        if !std::path::Path::new(&filename).exists() {
            eprintln!("❌ Download failed. PDB ID '{}' may not exist.", id);
            std::process::exit(1);
        }
        
        println!("✅ Downloaded {}", filename);
        filename
    } else {
        // Use provided file or default
        let default_path = "/Users/dishant/Desktop/Atom/benzene.xyz".to_string();
        if args.len() > 1 { args[1].clone() } else { default_path }
    };
    
    println!("Loading molecule from: {}", xyz_path);
    
    // Check extension
    let mut system = if xyz_path.ends_with(".pdb") {
         System::from_pdb(&xyz_path).expect("Failed to load PDB")
    } else if xyz_path.ends_with(".cif") {
         System::from_cif(&xyz_path).expect("Failed to load CIF")
    } else {
         System::from_xyz(&xyz_path).expect("Failed to load XYZ")
    };
    
    println!("Loaded {} atoms.", system.positions.len());
    
    // Apply Biological Assembly (if present)
    system.expand_assembly();
    
    // Validate Data
    if system.positions.iter().any(|p| p.is_nan()) {
        println!("🔥 FATAL: Loaded positions contain NaNs!");
    }
    if system.masses.iter().any(|&m| m <= 0.0) {
        println!("🔥 FATAL: Loaded masses contain 0.0 or negative!");
    }
        
    // Validate Velocities (Should be zero initialized)
    // Removed Maxwell-Boltzmann kick to ensure total stability for large proteins.
    // system.initialize_maxwell_boltzmann(300.0);
    if system.velocities.iter().any(|v| v.is_nan()) {
        println!("🔥 FATAL: Velocities contain NaNs after Maxwell-Boltzmann!");
    }

    // 2. No Force Field (Pure Visualization)
    // The user requested to remove force calculations to avoid "mess".
    // We define a NoOp force field to ensure forces remain ZERO.
    // (Passing None might default to LennardJones which explodes on raw PDB overlaps)
    
    struct NoOpForceField;
    impl ForceField for NoOpForceField {
        fn calculate_forces(&mut self, _system: &mut System, _grid: &SpatialGrid) {
            // Do nothing. Forces remain zero.
        }
    }

    println!("Starting Visualizer (Forces Disabled)...");
    println!("Controls: SPACE to Pause/Unpause. Mouse to Rotate/Pan. F to Focus. G to Pivot.");

    pollster::block_on(atom_viz::run(system, Some(Box::new(NoOpForceField)), std::collections::HashMap::new()));
}
