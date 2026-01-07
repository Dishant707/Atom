use atom_core::{System, SpatialGrid};
use glam::Vec3;

pub mod neural; // AI Hooks
pub mod ani;
pub mod hybrid;


// The AI Hook
pub trait ForceField {
    fn calculate_forces(&mut self, system: &mut System, grid: &SpatialGrid);
}

pub struct VelocityVerlet {
    pub dt: f32,
}

pub struct BerendsenThermostat {
    pub target_temp: f32,
    pub tau: f32, // Coupling constant
}

impl BerendsenThermostat {
    pub fn new(target_temp: f32, tau: f32) -> Self {
        Self { target_temp, tau }
    }
    
    pub fn apply(&self, system: &mut System, dt: f32) {
        let _kb = 0.001987; // kcal/(mol*K) - units need to match system. Simplified: using 1.0 for now for demo physics
        // Actually, let's stick to consistent units. 
        // KE = 0.5 * m * v^2
        // T = (2 * KE) / (3 * N * kB)
        
        // For this demo with mass=1.0 and arbitrary units:
        // Let's assume kB = 1.0 for simplicity of visual behavior
        
        let mut total_ke = 0.0;
        for (i, vel) in system.velocities.iter().enumerate() {
            let speed_sq = vel.length_squared();
            total_ke += 0.5 * system.masses[i] * speed_sq;
        }
        
        let n = system.positions.len() as f32;
        // avoid div by zero
        if n == 0.0 { return; }
        
        let current_temp = (2.0 * total_ke) / (3.0 * n); // Assuming kB=1
        
        // Lambda scaling factor
        // lambda = sqrt(1 + (dt/tau) * (T_target / T_current - 1))
        
        if current_temp < 1e-6 { return; } // Too cold to scale securely
        
        let scale = (1.0 + (dt / self.tau) * (self.target_temp / current_temp - 1.0)).sqrt();
        
        for vel in &mut system.velocities {
            *vel *= scale;
        }
    }
}


impl VelocityVerlet {
    pub fn new(dt: f32) -> Self {
        Self { dt }
    }

    pub fn integrate(&self, system: &mut System, grid: &SpatialGrid, force_field: &mut dyn ForceField) {
        let dt = self.dt;
        let dt_sq_half = 0.5 * dt * dt;
        let dt_half = 0.5 * dt;

        let num_atoms = system.positions.len();

        // 1. Update positions
        for i in 0..num_atoms {
            let inv_mass = 1.0 / system.masses[i];
            let acc = system.forces[i] * inv_mass;
            
            system.positions[i] += system.velocities[i] * dt + acc * dt_sq_half;
            system.velocities[i] += acc * dt_half;
            
            // Boundary (Simple Periodic Wrapper)
            if system.positions[i].x < 0.0 { system.positions[i].x += system.box_size.x; }
            if system.positions[i].x > system.box_size.x { system.positions[i].x -= system.box_size.x; }
            if system.positions[i].y < 0.0 { system.positions[i].y += system.box_size.y; }
            if system.positions[i].y > system.box_size.y { system.positions[i].y -= system.box_size.y; }
            if system.positions[i].z < 0.0 { system.positions[i].z += system.box_size.z; }
            if system.positions[i].z > system.box_size.z { system.positions[i].z -= system.box_size.z; }
        }

        // 2. Calculate Forces via Interface (AI or Classic)
        for force in system.forces.iter_mut() {
            *force = Vec3::ZERO;
        }
        
        force_field.calculate_forces(system, grid);

        // 3. Final velocity update
        for i in 0..num_atoms {
            let inv_mass = 1.0 / system.masses[i];
            let acc = system.forces[i] * inv_mass;
            system.velocities[i] += acc * dt_half;
        }
    }
}

pub struct LennardJones {
    pub epsilon: f32,
    pub sigma: f32,
}

impl LennardJones {
    pub fn new(epsilon: f32, sigma: f32) -> Self {
        Self { epsilon, sigma }
    }
}

impl ForceField for LennardJones {
    fn calculate_forces(&mut self, system: &mut System, grid: &SpatialGrid) {
        let epsilon = self.epsilon;
        let sigma = self.sigma;
        let cutoff = 2.5 * sigma;
        let cutoff_sq = cutoff * cutoff;

        let dims = grid.dimensions;
        
        // This is where we plug in the AI later:
        // fn calculate_forces(...) {
        //   let tensor = system_to_tensor(system);
        //   let forces = model.forward(tensor);
        //   system.forces = forces;
        // }

        for x in 0..dims.0 {
            for y in 0..dims.1 {
                for z in 0..dims.2 {
                    let cell_idx = x + y * dims.0 + z * dims.0 * dims.1;
                    let atoms_in_cell = &grid.cells[cell_idx];
                    
                    if atoms_in_cell.is_empty() { continue; }

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                let nz = z as isize + dz;

                                if nx >= 0 && nx < dims.0 as isize &&
                                   ny >= 0 && ny < dims.1 as isize &&
                                   nz >= 0 && nz < dims.2 as isize {
                                       
                                    let n_idx = nx as usize + ny as usize * dims.0 + nz as usize * dims.0 * dims.1;
                                    let neighbor_atoms = &grid.cells[n_idx];

                                    for &i_idx in atoms_in_cell {
                                        for &j_idx in neighbor_atoms {
                                            let i = i_idx as usize;
                                            let j = j_idx as usize;
                                            
                                            if i >= j { continue; }

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

    }
}

pub struct HarmonicBond {
    pub k: f32,
}

impl HarmonicBond {
    pub fn new(k: f32) -> Self {
        Self { k }
    }
    
    fn get_covalent_radius(z: u8) -> f32 {
        match z {
            1 => 0.31, // H
            6 => 0.76, // C
            7 => 0.71, // N
            8 => 0.66, // O
            9 => 0.57, // F
            15 => 1.07, // P
            16 => 1.05, // S
            17 => 1.02, // Cl
            _ => 0.75, // Generic
        }
    }
}

impl ForceField for HarmonicBond {
    fn calculate_forces(&mut self, system: &mut System, _grid: &SpatialGrid) {
        // Iterate over all bonds in the system
        for &(i_idx, j_idx, _order) in &system.bonds {
            let i = i_idx as usize;
            let j = j_idx as usize;
            
            if i >= system.positions.len() || j >= system.positions.len() { continue; }
            
            let pos_i = system.positions[i];
            let pos_j = system.positions[j];
            
            let mut r_vec = pos_j - pos_i;
            
            // Minimum Image Convention (simple)
            if r_vec.x > system.box_size.x * 0.5 { r_vec.x -= system.box_size.x; }
            else if r_vec.x < -system.box_size.x * 0.5 { r_vec.x += system.box_size.x; }
            if r_vec.y > system.box_size.y * 0.5 { r_vec.y -= system.box_size.y; }
            else if r_vec.y < -system.box_size.y * 0.5 { r_vec.y += system.box_size.y; }
            if r_vec.z > system.box_size.z * 0.5 { r_vec.z -= system.box_size.z; }
            else if r_vec.z < -system.box_size.z * 0.5 { r_vec.z += system.box_size.z; }
            
            let dist = r_vec.length();
            if dist < 1e-4 { continue; } // Singularity check
            
            let z1 = system.atomic_numbers[i];
            let z2 = system.atomic_numbers[j];
            let r0 = Self::get_covalent_radius(z1) + Self::get_covalent_radius(z2);
            
            // F = -k * (r - r0)
            let displacement = dist - r0;
            let force_mag = -self.k * displacement;
            
            let force_dir = r_vec / dist;
            let force = force_dir * force_mag;
            
            system.forces[i] -= force;
            system.forces[j] += force;
        }
    }
}

pub struct MolecularForceField {
    pub bonds: HarmonicBond,
    pub vdw: LennardJones,
}

impl MolecularForceField {
    pub fn new() -> Self {
        Self {
            bonds: HarmonicBond::new(300.0), // k = 300
            vdw: LennardJones::new(0.1, 1.0),
        }
    }
}

impl ForceField for MolecularForceField {
    fn calculate_forces(&mut self, system: &mut System, grid: &SpatialGrid) {
        self.bonds.calculate_forces(system, grid);
        self.vdw.calculate_forces(system, grid);
    }
}
