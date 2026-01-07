use crate::{ForceField, ani::AniEnsemble, LennardJones};
use atom_core::{System, SpatialGrid};
use glam::Vec3;

pub struct HybridForceField {
    pub ani: AniEnsemble,
    pub lj: LennardJones,
    pub qm_region: Vec<usize>,
}

impl HybridForceField {
    pub fn new(ani: AniEnsemble, lj: LennardJones, qm_region: Vec<usize>) -> Self {
        Self { ani, lj, qm_region }
    }
    
    pub fn set_qm_region(&mut self, indices: Vec<usize>) {
        self.qm_region = indices;
    }
}

impl ForceField for HybridForceField {
    fn calculate_forces(&mut self, system: &mut System, grid: &SpatialGrid) {
        // 1. Calculate MM Forces (Lennard-Jones) with QM Exclusion
        // We calculate LJ for:
        // - MM-MM pairs
        // - QM-MM pairs (Interaction/Embedding)
        // We SKIP LJ for:
        // - QM-QM pairs (ANI handles these entirely, including repulsion)

        let epsilon = self.lj.epsilon;
        let sigma = self.lj.sigma;
        let cutoff = 2.5 * sigma;
        let cutoff_sq = cutoff * cutoff;
        let dim = grid.dimensions;

        // Use a Set for fast lookup of QM status
        // (Optimisation: Pre-calculate this or store it better if resizing)
        let qm_set: std::collections::HashSet<usize> = self.qm_region.iter().cloned().collect();

        // Standard Grid Loop (copied from LennardJones implementation)
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                for z in 0..dim.2 {
                    let cell_idx = x + y * dim.0 + z * dim.0 * dim.1;
                    let atoms_in_cell = &grid.cells[cell_idx];
                    
                    if atoms_in_cell.is_empty() { continue; }

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                let nz = z as isize + dz;

                                if nx >= 0 && nx < dim.0 as isize &&
                                   ny >= 0 && ny < dim.1 as isize &&
                                   nz >= 0 && nz < dim.2 as isize {
                                       
                                    let n_idx = nx as usize + ny as usize * dim.0 + nz as usize * dim.0 * dim.1;
                                    let neighbor_atoms = &grid.cells[n_idx];

                                    for &i_idx in atoms_in_cell {
                                        for &j_idx in neighbor_atoms {
                                            let i = i_idx as usize;
                                            let j = j_idx as usize;
                                            
                                            if i >= j { continue; }

                                            // CRITICAL FIX: Skip if BOTH are QM atoms
                                            if qm_set.contains(&i) && qm_set.contains(&j) {
                                                continue;
                                            }

                                            let mut r_vec = system.positions[j] - system.positions[i];
                                            
                                            // Minimum Image Convention
                                            if r_vec.x > system.box_size.x * 0.5 { r_vec.x -= system.box_size.x; }
                                            else if r_vec.x < -system.box_size.x * 0.5 { r_vec.x += system.box_size.x; }
                                            if r_vec.y > system.box_size.y * 0.5 { r_vec.y -= system.box_size.y; }
                                            else if r_vec.y < -system.box_size.y * 0.5 { r_vec.y += system.box_size.y; }
                                            if r_vec.z > system.box_size.z * 0.5 { r_vec.z -= system.box_size.z; }
                                            else if r_vec.z < -system.box_size.z * 0.5 { r_vec.z += system.box_size.z; }
                                            
                                            let r2 = r_vec.length_squared();

                                            if r2 < cutoff_sq && r2 > 1e-6 {
                                                let inv_r2 = 1.0 / r2;
                                                let sig2 = sigma * sigma;
                                                let sr2 = sig2 * inv_r2;
                                                let sr6 = sr2 * sr2 * sr2;
                                                let sr12 = sr6 * sr6;
                                                
                                                let force_mag = (24.0 * epsilon * inv_r2) * (2.0 * sr12 - sr6);
                                                let force = r_vec * force_mag;

                                                system.forces[i] -= force;
                                                system.forces[j] += force;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Add ANI Forces for QM atoms (QM-QM interactions)
        // This calculates Bond, Angle, Dihedral, and QM-VdW terms.
        let _ = self.ani.calculate_subset_forces(system, &self.qm_region);
    }
}
