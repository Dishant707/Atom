use atom_core::System;
use atom_physics::ani::AniEnsemble;
use atom_physics::hybrid::HybridForceField;
use atom_physics::LennardJones;
use atom_viz::run;
use glam::Vec3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filename = if args.len() > 1 { &args[1] } else { "1crn.pdb" };

    println!("🔥 Loading System: {}", filename);
    
    // Load System
    let mut system = if filename.ends_with(".pdb") {
        System::from_pdb(filename).expect("Failed to load PDB")
    } else {
        System::from_xyz(filename).expect("Failed to load XYZ")
    };
    
    println!("✅ System loaded with {} atoms.", system.positions.len());
    println!("🧪 Atomic Numbers: {:?}", system.atomic_numbers);
    system.box_size = glam::Vec3::new(100.0, 100.0, 100.0); // Ensure no PBC overlap

    // 1. Define QM Region (The "Reactive Site")
    // If small molecule, pick first half. If large, pick specific range.
    let n_atoms = system.positions.len();
    let qm_indices: Vec<usize> = if n_atoms > 100 {
        // For protein, pick start 0-50
        (0..50).collect() 
    } else {
        // For small molecule (e.g. Benzene), pick first half
        (0..n_atoms/2).collect()
    };
    println!("🔬 Defined QM Region: {} atoms", qm_indices.len());

    // 2. Initialize ANI-2x (Quantum Engine)
    println!("🧠 Initializing ANI-2x (Active Site)...");
    let model_dir = "/Users/dishant/Downloads/ani2xr_onnx_models";
    let ani = AniEnsemble::new(model_dir).expect("Failed to load ANI models");

    // 3. Initialize Lennard-Jones (Background Physics)
    println!("🌎 Initializing Classical Force Field (Background)...");
    // Standard protein parameters (approximate)
    // Epsilon ~ 0.1 kcal/mol, Sigma ~ 3.0 Angstroms
    let lj = LennardJones::new(0.1, 3.4);

    // 4. Create Hybrid Engine
    let force_field = HybridForceField::new(ani, lj, qm_indices.clone());
    
    // 5. Inject Energy
    system.initialize_maxwell_boltzmann(50.0);

    // 6. Setup Visualization
    // We want to highlight the QM atoms to show the "Active Zone"
    let mut custom_colors = std::collections::HashMap::new();
    for &idx in &qm_indices {
        // Highlight QM atoms in Bright Green/Cyan
        custom_colors.insert(idx, [0.0, 1.0, 0.5]); 
    }
    
    // Run Visualization
    println!("🚀 Starting Hybrid QM/MM Simulation...");
    println!("   QM Region: Green (ANI-2x Physics)");
    println!("   MM Region: CPK Colors (Classical Physics)");
    
    pollster::block_on(run(system, Some(Box::new(force_field)), custom_colors));
}
