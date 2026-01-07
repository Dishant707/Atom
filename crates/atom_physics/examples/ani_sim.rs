use atom_core::{System, SpatialGrid};
use atom_physics::{VelocityVerlet, ForceField};
use atom_physics::neural::NeuralPotential;
use glam::Vec3;

fn main() -> anyhow::Result<()> {
    // 1. Setup System: Methane (CH4)
    // C at 0, H at tetra vertices
    let mut system = System::new(5);
    system.box_size = Vec3::new(20.0, 20.0, 20.0);
    
    // C
    system.add_atom(Vec3::new(10.0, 10.0, 10.0), 12.01, 6, "A".to_string(), "MET".to_string(), 1, atom_core::SecondaryStructure::Unknown, "C".to_string());
    // H 1 (+0.63, +0.63, +0.63)
    system.add_atom(Vec3::new(10.63, 10.63, 10.63), 1.008, 1, "A".to_string(), "MET".to_string(), 1, atom_core::SecondaryStructure::Unknown, "H".to_string());
    // H 2 (-0.63, -0.63, +0.63)
    system.add_atom(Vec3::new(9.37, 9.37, 10.63), 1.008, 1, "A".to_string(), "MET".to_string(), 1, atom_core::SecondaryStructure::Unknown, "H".to_string());
    // H 3 (-0.63, +0.63, -0.63)
    system.add_atom(Vec3::new(9.37, 10.63, 9.37), 1.008, 1, "A".to_string(), "MET".to_string(), 1, atom_core::SecondaryStructure::Unknown, "H".to_string());
    // H 4 (+0.63, -0.63, -0.63)
    system.add_atom(Vec3::new(10.63, 9.37, 9.37), 1.008, 1, "A".to_string(), "MET".to_string(), 1, atom_core::SecondaryStructure::Unknown, "H".to_string());

    // Initial Kick to see vibration
    system.velocities[1].x += 0.01;

    // 2. Setup AI Brain
    let model_path = "/Users/dishant/Desktop/Atom/ani2x_complete.onnx";
    println!("Initializing Neural Potential...");
    let mut brain = NeuralPotential::new(model_path)?;
    
    // 3. Setup Integrator
    let integrator = VelocityVerlet::new(0.5); // dt = 0.5 fs (Atomic units usually require small steps)
    
    let mut grid = SpatialGrid::new(system.box_size, 5.0);

    println!("Starting Simulation...");
    println!("Step | C Position | H1 Dist | Total Force on H1");
    
    for i in 0..50 {
        grid.insert(&system);
        
        // Use the AI!
        integrator.integrate(&mut system, &grid, &mut brain);
        
        // Metrics
        let c_pos = system.positions[0];
        let h1_pos = system.positions[1];
        let dist = c_pos.distance(h1_pos);
        let f_h1 = system.forces[1].length();
        
        println!("{:4} | {:.4}, {:.4}, {:.4} | {:.4} | {:.6}", 
            i, c_pos.x, c_pos.y, c_pos.z, dist, f_h1);
    }
    
    Ok(())
}
