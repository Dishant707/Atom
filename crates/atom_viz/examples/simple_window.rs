use atom_core::System;

fn main() {
    // 10x10x10 grid of atoms
    let mut system = System::new(1000);
    // Box size larger to contain them
    system.box_size = glam::Vec3::new(30.0, 30.0, 30.0);
    
    let spacing = 1.2;
    for x in 0..10 {
        for y in 0..10 {
            for z in 0..10 {
                let pos = glam::Vec3::new(
                    (x as f32) * 2.0 + 5.0, // Start from 5.0
                    (y as f32) * 2.0 + 5.0,
                    (z as f32) * 2.0 + 5.0
                );
                system.add_atom(pos, 1.0, 6, "A".to_string(), "UNK".to_string(), 0, atom_core::SecondaryStructure::Unknown, "C".to_string()); // Carbon
            }
        }
    }
    
    // Create a slight perturbation to make it interesting (unbalanced forces)
    system.positions[555] += glam::Vec3::new(0.5, 0.0, 0.0);

    pollster::block_on(atom_viz::run(system, None, std::collections::HashMap::new()));
}
