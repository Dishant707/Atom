use atom_core::{System, SpatialGrid};
use atom_physics::{ForceField, MolecularForceField, VelocityVerlet};
use glam::Vec3;

fn main() {
    let mut system = System::new(2);
    
    // Create C-C pair at 1.54 A distance
    system.positions.push(Vec3::new(0.0, 0.0, 0.0));
    system.positions.push(Vec3::new(1.54, 0.0, 0.0)); // Perfect equilibrium
    
    system.atomic_numbers.push(6); // C
    system.atomic_numbers.push(6); // C
    
    system.masses.push(12.0);
    system.masses.push(12.0);
    
    system.velocities.push(Vec3::ZERO);
    system.velocities.push(Vec3::ZERO);
    system.forces.push(Vec3::ZERO);
    system.forces.push(Vec3::ZERO);
    
    system.bonds.push((0, 1)); // Add bond
    
    let grid = SpatialGrid::new(system.box_size, 2.5);
    let mut ff = MolecularForceField::new();
    let integrator = VelocityVerlet::new(0.01); // 10fs? 0.01 is small in whatever units.
    
    println!("Step 0: Dist = {:.4}", (system.positions[1] - system.positions[0]).length());
    
    for i in 1..11 {
        // Integrate
        // Note: simple integration without grid updates just for test
        // ff.calculate_forces(&mut system, &grid); 
        // integrator handles full step including force calc
        // But integrator needs &mut ForceField
        
        // We can't use integrator directly easily if signatures mismatch or ownership issues
        // Let's just manually call calc forces.
        
        ff.calculate_forces(&mut system, &grid); // Grid is empty but force field (bond) doesn't use grid for bonds.
        // Update pos (Euler for simplicity in test)
        for k in 0..2 {
            let acc = system.forces[k] / system.masses[k];
            system.positions[k] += system.velocities[k] * 0.1 + 0.5 * acc * 0.01;
            system.velocities[k] += acc * 0.1;
            system.forces[k] = Vec3::ZERO; // Reset
        }
        
        println!("Step {}: Dist = {:.4}, Force0 = {:.4}", i, (system.positions[1] - system.positions[0]).length(), system.forces[0].length());
    }
}
