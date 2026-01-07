use atom_core::System;
use atom_physics::ani::AniEnsemble;
use glam::Vec3;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧬 ANI-2x Ensemble Test: Crambin Protein");
    println!("==========================================\n");
    
    // Load models
    let models_dir = "/Users/dishant/Downloads/ani2xr_onnx_models";
    println!("Loading ANI-2x models from: {}", models_dir);
    
    let mut ani = AniEnsemble::new(models_dir)?;
    println!("✅ Models loaded successfully\n");
    
    // Create a simple test system - Methane (CH4)
    // This is simpler than the full protein for initial testing
    println!("Creating test system: Methane (CH4)");
    
    let mut system = System {
        positions: vec![
            Vec3::new(0.0000, 0.0000, 0.0000),  // C
            Vec3::new(0.6300, 0.6300, 0.6300),  // H
            Vec3::new(-0.6300,-0.6300, 0.6300),  // H
            Vec3::new(-0.6300, 0.6300,-0.6300),  // H
            Vec3::new(0.6300,-0.6300,-0.6300),  // H
        ],
        velocities: vec![Vec3::ZERO; 5],
        forces: vec![Vec3::ZERO; 5],
        masses: vec![12.01, 1.008, 1.008, 1.008, 1.008],
        atomic_numbers: vec![6, 1, 1, 1, 1], // C, H, H, H, H
        box_size: Vec3::new(100.0, 100.0, 100.0),
        bonds: vec![],
        ids: vec![0, 1, 2, 3, 4],
        secondary_structure: vec![],
        atom_names: vec![],
        ..Default::default()
    };
    
    println!("Atom count: {}", system.positions.len());
    println!("Species: C + 4H\n");
    
    // Calculate forces
    println!("Computing forces...");
    ani.calculate_forces(&mut system)?;
    
    println!("\n📊 Results:");
    println!("===========");
    for (i, &force) in system.forces.iter().enumerate() {
        let atom_type = if system.atomic_numbers[i] == 6 { "C" } else { "H" };
        println!("Atom {} ({}): F = [{:>10.6}, {:>10.6}, {:>10.6}] Hartree/Å", 
            i, atom_type, force.x, force.y, force.z);
    }
    
    // Calculate total force (should be ~0 due to Newton's 3rd law)
    let total_force: Vec3 = system.forces.iter().sum();
    println!("\nTotal force: [{:.6}, {:.6}, {:.6}] (should be ~0)", 
        total_force.x, total_force.y, total_force.z);
    
    println!("\n✅ Test complete!");
    Ok(())
}
