use atom_core::System;

fn main() {
    let system = System::from_pdb("test_data/1crn.pdb"); // Example placeholder
    match system {
        Ok(sys) => println!("Loaded {} atoms", sys.positions.len()),
        Err(e) => println!("Error loading PDB: {}", e),
    }
    
    // Demonstrate manual creation
    let mut sys = System::new(10);
    // system.add_atom(Vec3::new(0.000, 0.000, 0.000), 1.0, 1, "A".to_string(), "UNK".to_string(), 0, atom_core::SecondaryStructure::Unknown, "H".to_string());
    println!("Created manual system with {} atoms", sys.positions.len());
}
