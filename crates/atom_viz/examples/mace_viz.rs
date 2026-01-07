use atom_core::{System, SpatialGrid};
use atom_physics::mace::MacePotential;
use atom_physics::ForceField;
use glam::Vec3;

fn main() {
    // 1. Setup System: Benzene
    let xyz_path = "/Users/dishant/Desktop/Atom/benzene.xyz";
    println!("Loading molecule from: {}", xyz_path);
    
    let mut system = System::from_xyz(xyz_path)
        .expect("Failed to load XYZ file");
        
    // Initial Kick to see vibration
    system.initialize_maxwell_boltzmann(300.0);

    // 2. Setup AI Brain (MACE)
    // Converted using mace_create_lammps_model
    let model_path = "/Users/dishant/Desktop/Atom/2023-12-03-mace-128-L1_epoch-199.model-lammps.pt"; 
    
    // MACE cutoff is usually 5.0 - 6.0 Angstroms
    let cutoff = 5.0; 
    
    println!("Initializing MACE Potential from {}...", model_path);
    
    match MacePotential::new(model_path, cutoff) {
        Ok(brain) => {
             println!("MACE Loaded! Starting Visualizer...");
             println!("Controls: SPACE to Pause/Unpause. Mouse to Rotate/Pan. F to Focus. G to Pivot.");
             
             // Cast to generic ForceField
             let force_field: Box<dyn ForceField> = Box::new(brain);
             
             pollster::block_on(atom_viz::run(system, Some(force_field), std::collections::HashMap::new()));
        },
        Err(e) => {
            eprintln!("Failed to load MACE model: {}", e);
            eprintln!("NOTE: If the error mentions 'Pkl' or 'pickle', you need to convert the .model file to .pt (TorchScript) using Python!");
        }
    }
}
