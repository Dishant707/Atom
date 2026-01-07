use crate::ForceField;
use atom_core::{System, SpatialGrid};
use atom_core::potential::Potential;
use glam::Vec3;
use anyhow::Result;

pub struct NeuralPotential {
    pub potential: Potential,
    pub scaling_factor: f32, // To convert Hartree/Angstrom to internal units if needed
}

impl NeuralPotential {
    pub fn new(model_path: &str) -> Result<Self> {
        let potential = Potential::new(model_path)?;
        Ok(Self { 
            potential,
            scaling_factor: 1.0 // ANI implies Hartree/Angstrom. Atom usually needs checking.
            // If Atom uses generic units, we might need a large scalar.
            // 1 Hartree ≈ 627.5 kcal/mol
            // If internal units are simple, keeping 1.0 is fine for "relative" physics.
        })
    }
}

impl ForceField for NeuralPotential {
    fn calculate_forces(&mut self, system: &mut System, _grid: &SpatialGrid) {
        // Convert System to Neural Input
        // 1. Species (Atomic Number) -> ANI Index Map
        // Map: H(1)->0, C(6)->1, N(7)->2, O(8)->3, S(16)->4, F(9)->5, Cl(17)->6
        let mut species = Vec::with_capacity(system.positions.len());
        let mut coords = Vec::with_capacity(system.positions.len() * 3);

        for (i, &z) in system.atomic_numbers.iter().enumerate() {
            let idx = match z {
                1 => 0,
                6 => 1,
                7 => 2,
                8 => 3,
                16 => 4,
                9 => 5,
                17 => 6,
                _ => 1, // Default to Carbon if unknown? Or panic?
            };
            species.push(idx);
            
            let pos = system.positions[i];
            coords.push(pos.x);
            coords.push(pos.y);
            coords.push(pos.z);
        }

        // 2. Call Brain
        match self.potential.compute(&species, &coords) {
            Ok((_energy, forces_flat)) => {
                // 3. Apply Forces
                for (i, chunk) in forces_flat.chunks(3).enumerate() {
                    if i < system.forces.len() {
                        // ANI outputs Force = -Gradient.
                        // Potential::compute already returns F = -grad.
                        // So we just add it to system.forces.
                        let f = Vec3::new(chunk[0], chunk[1], chunk[2]);
                        
                        // Scale units if necessary (Hartree/A -> Internal)
                        // For visuals, raw Hartree/A is often too strong/weak depending on mass=1.0 assumption.
                        // Mass=12.01 (Carbon) vs Force~0.1. a = F/m ~ 0.008 A/fs^2.
                        // That's reasonable.
                        
                        system.forces[i] += f;
                    }
                }
            },
            Err(e) => {
                eprintln!("Neural Potential Error: {}", e);
            }
        }
    }
}
