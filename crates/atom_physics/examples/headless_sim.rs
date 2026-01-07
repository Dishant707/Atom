use atom_core::{System, SpatialGrid};
use atom_physics::VelocityVerlet;
use glam::Vec3;

fn main() {
    // 1. Setup System
    // Create 2 atoms close to each other
    let mut system = System::new(10);
    // Box size 10x10x10
    system.box_size = Vec3::new(10.0, 10.0, 10.0);
    
    // Atom 1 at center
    system.add_atom(Vec3::new(5.0, 5.0, 5.0), 1.0, 6, "A".to_string(), "UNK".to_string(), 0, atom_core::SecondaryStructure::Unknown, "C".to_string());
    // Atom 2 slightly offset (distance 1.1, sigma=1.0 so should be near equilibrium/small force)
    system.add_atom(Vec3::new(6.1, 5.0, 5.0), 1.0, 6, "A".to_string(), "UNK".to_string(), 0, atom_core::SecondaryStructure::Unknown, "C".to_string());

    // 2. Setup Grid
    let mut grid = SpatialGrid::new(system.box_size, 2.5); // 2.5 cut-off radius
    
    // 3. Setup Integrator and Force Field
    let integrator = VelocityVerlet::new(0.01); // dt = 0.01
    let mut force_field = atom_physics::LennardJones::new(0.1, 1.0); // epsilon=0.1, sigma=1.0

    println!("Initial Pos 1: {}", system.positions[0]);
    println!("Initial Pos 2: {}", system.positions[1]);

    // 4. Run Loop
    for i in 0..10 {
        grid.insert(&system);
        integrator.integrate(&mut system, &grid, &mut force_field);
        
        println!("Step {}: Pos 1: {}, Pos 2: {}, Force 1: {}", 
            i, system.positions[0], system.positions[1], system.forces[0]);
    }
}
