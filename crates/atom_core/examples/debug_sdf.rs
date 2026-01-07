use atom_core::System;
use glam::Vec3;

fn main() {
    let path = "Aspirin.sdf";
    println!("Loading {}...", path);
    
    let mut system = System::from_sdf(path).expect("Failed to load SDF");
    println!("Loaded system with {} atoms.", system.positions.len());
    println!("Box size: {}", system.box_size);
    
    // Check first few atoms
    for i in 0..std::cmp::min(5, system.positions.len()) {
        println!("Atom {}: {} pos={:?} mass={}", 
            i, system.atom_names[i], system.positions[i], system.masses[i]);
    }
    
    // Simulate simple integration steps (Velocity Verlet basic)
    let dt = 1.0; // 1fs
    println!("\nSimulating...");
    
    for step in 0..10 {
        // Mock force calculation (Gravity towards center? or just check if forces are zero)
        // In the real app, the viz crate sets up the integrator. 
        // We just want to see if coordinates look sane initialy.
        
        // Recalculate Min/Max
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for &p in &system.positions {
            min = min.min(p);
            max = max.max(p);
        }
        let spread = max - min;
        println!("Step {}: Spread = {:.4}, Center = {:?}", step, spread.length(), (min+max)*0.5);
        
        // If spread < 0.1, it collapsed.
        if spread.length() < 0.1 {
            println!("COLLAPSED!");
            break;
        }
    }
}
