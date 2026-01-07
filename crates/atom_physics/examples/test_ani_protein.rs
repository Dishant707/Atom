use atom_core::System;
use atom_physics::ani::AniEnsemble;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧬 ANI-2x Test: Crambin Protein (1CRN)");
    println!("=========================================\n");
    
    // Load protein structure
    let pdb_path = "/Users/dishant/Desktop/Atom/1crn.pdb";
    println!("Loading protein from: {}", pdb_path);
    
    let mut system = System::from_pdb(pdb_path)
        .expect("Failed to load PDB file");
    
    println!("✅ Loaded {} atoms", system.positions.len());
    println!("   Box size: {:.2} x {:.2} x {:.2} Å\n", 
        system.box_size.x, system.box_size.y, system.box_size.z);
    
    // Count element types
    let mut element_counts = std::collections::HashMap::new();
    for &z in &system.atomic_numbers {
        *element_counts.entry(z).or_insert(0) += 1;
    }
    
    println!("Element composition:");
    for (&z, &count) in &element_counts {
        let symbol = match z {
            1 => "H", 6 => "C", 7 => "N", 8 => "O", 
            15 => "P", 16 => "S", _ => "?"
        };
        println!("  {}: {}", symbol, count);
    }
    
    // Load ANI models
    let models_dir = "/Users/dishant/Downloads/ani2xr_onnx_models";
    println!("\nLoading ANI-2x models from: {}", models_dir);
    
    let mut ani = AniEnsemble::new(models_dir)?;
    println!("✅ Models loaded\n");
    
    // Compute forces
    println!("Computing forces on {} atoms...", system.positions.len());
    println!("(This may take a moment for large proteins)\n");
    
    ani.calculate_forces(&mut system)?;
    
    // Analyze results
    println!("📊 Force Analysis:");
    println!("==================");
    
    let mut max_force = 0.0_f32;
    let mut max_force_idx = 0;
    let mut total_force = glam::Vec3::ZERO;
    
    for (i, &force) in system.forces.iter().enumerate() {
        let mag = force.length();
        total_force += force;
        
        if mag > max_force {
            max_force = mag;
            max_force_idx = i;
        }
    }
    
    println!("Total atoms:     {}", system.positions.len());
    println!("Max force:       {:.6} Hartree/Å (atom {})", max_force, max_force_idx);
    println!("Total force:     [{:.6}, {:.6}, {:.6}]", 
        total_force.x, total_force.y, total_force.z);
    println!("Force magnitude: {:.6} (should be ~0)", total_force.length());
    
    // Show forces on first few atoms
    println!("\nSample forces (first 10 atoms):");
    for i in 0..10.min(system.positions.len()) {
        let force = system.forces[i];
        let atom_type = match system.atomic_numbers[i] {
            1 => "H", 6 => "C", 7 => "N", 8 => "O", 
            15 => "P", 16 => "S", _ => "?"
        };
        println!("  Atom {:3} ({}): F = [{:>10.6}, {:>10.6}, {:>10.6}] |F| = {:.6}", 
            i, atom_type, force.x, force.y, force.z, force.length());
    }
    
    println!("\n✅ Calculation complete!");
    
    Ok(())
}
