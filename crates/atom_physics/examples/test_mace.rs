use atom_physics::mace::MacePotential;
use atom_physics::ForceField;
use atom_core::{System, SpatialGrid};
use glam::Vec3;
use tch::{Tensor, Kind, Device, IValue};

fn main() {
    println!("--- Headless MACE Test ---");
    let model_path = "/Users/dishant/Desktop/Atom/2023-12-03-mace-128-L1_epoch-199.model-lammps.pt";
    let cutoff = 5.0;

    // 1. Load Model
    println!("Loading model from {}...", model_path);
    let mut potential = match MacePotential::new(model_path, cutoff) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };
    println!("Model loaded.");

    // 2. Create Dummy System (2 atoms)
    // C-C bond length approx 1.54 A
    let mut system = System {
        positions: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.54, 0.0, 0.0)],
        atomic_numbers: vec![6, 6], // Two Carbons
        forces: vec![Vec3::ZERO, Vec3::ZERO],
        velocities: vec![Vec3::ZERO, Vec3::ZERO],
        masses: vec![12.0, 12.0],
        box_size: Vec3::new(10.0, 10.0, 10.0), // Box
        atom_names: vec!["C".to_string(), "C".to_string()],
        ..Default::default() // other fields
    };

    // 3. Create Dummy Grid
    let mut grid = SpatialGrid::new(system.box_size, cutoff); 
    grid.insert(&system);
    
    println!("Calculating forces...");
    potential.calculate_forces(&mut system, &grid);
    
    println!("Forces on Atom 0: {:?}", system.forces[0]);
    println!("Forces on Atom 1: {:?}", system.forces[1]);
    
    if system.forces[0] == Vec3::ZERO {
         println!("WARNING: Forces are ZERO. Neighbor list might be empty or model returned zero.");
    } else {
         println!("SUCCESS: Non-zero forces computed!");
    }
}
