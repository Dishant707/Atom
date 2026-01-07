use glam::Vec3;
use std::collections::HashMap;
use std::f32::consts::PI;
use anyhow::{Result, Context};
use ort::session::Session;
use atom_core::System;
use ort::value::Value;
use ndarray::Array2;

// --- Constants (Extracted from ani2xr.pt) ---
const R_CR: f32 = 5.2000;
const R_CA: f32 = 3.5000;
// const HARTREE_TO_EV: f32 = 27.211386; // Results in explosion (Forces > 100 eV/A)
const HARTREE_TO_EV: f32 = 0.043363; // kcal/mol -> eV (Empirically stable ~ 0.14 eV/A)

const RADIAL_ETA: f32 = 19.7;
const RADIAL_SHIFTS: [f32; 16] = [
    0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375, 2.5125, 2.78125, 
    3.05, 3.31875, 3.5875, 3.85625, 4.125, 4.39375, 4.6625, 4.93125
];

const ANGULAR_ETA: f32 = 12.5;
const ANGULAR_ZETA: f32 = 14.1;
const ANGULAR_SHIFTS: [f32; 8] = [
    0.9, 1.225, 1.55, 1.875, 2.2, 2.525, 2.85, 3.175
];
const ANGULAR_SECTIONS: [f32; 4] = [
    0.3926991, 1.1780972, 1.9634954, 2.7488935
];

// Map AtomicNumber -> Index 0..6
const SPECIES_MAP: [(i64, usize); 7] = [
    (1, 0), (6, 1), (7, 2), (8, 3), (16, 4), (9, 5), (17, 6)
]; 
// H, C, N, O, S, F, Cl

pub struct AniEnsemble {
    models: HashMap<i64, Session>,
    species_to_idx: HashMap<i64, usize>,
    // Repulsion XTB Parameters (7x7 matrices)
    y_ab: [[f32; 7]; 7],
    sqrt_alpha_ab: [[f32; 7]; 7],
    k_rep_ab: [[f32; 7]; 7],
}


impl AniEnsemble {
    pub fn new(models_dir: &str) -> Result<Self> {
        let mut models = HashMap::new();
        let species_info = [
            (1, "H"), (6, "C"), (7, "N"), (8, "O"), 
            (16, "S"), (9, "F"), (17, "Cl")
        ];

        let mut species_to_idx = HashMap::new();
        for (z, idx) in SPECIES_MAP.iter() {
            species_to_idx.insert(*z, *idx);
        }

        for (z, sym) in species_info.iter() {
            let path = format!("{}/ani2xr_{}.onnx", models_dir, sym);
            // Verify file exists
            if std::path::Path::new(&path).exists() {
                let session = Session::builder()?
                    .with_execution_providers([
                        ort::execution_providers::CPUExecutionProvider::default().build(),
                    ])?
                    .commit_from_file(&path)
                    .context(format!("Failed to load model for {}", sym))?;
                models.insert(*z, session);
            } else {
                 eprintln!("Warning: Model for {} not found at {}", sym, path);
            }
        }

        // Repulsion XTB Parameters (Extracted from ani2xr.pt)
        #[rustfmt::skip]
        let y_ab = [
            [1.2218827, 4.6769834, 5.7950983, 6.394023, 7.7614665, 16.575392, 19.181948],
            [4.6769834, 17.902021, 22.181816, 24.474312, 29.708456, 63.445396, 73.42247],
            [5.7950983, 22.181816, 27.48477, 30.325325, 36.810783, 78.61314, 90.9754],
            [6.394023, 24.474312, 30.325325, 33.459454, 40.615185, 86.737816, 100.37772],
            [7.7614665, 29.708456, 36.810783, 40.615185, 49.30126, 105.28781, 121.84479],
            [16.575392, 63.445396, 78.61314, 86.737816, 105.28781, 224.8527, 260.2118],
            [19.181948, 73.42247, 90.9754, 100.37772, 121.84479, 260.2118, 301.13126],
        ];

        #[rustfmt::skip]
        let sqrt_alpha_ab = [
            [2.213717, 1.661913, 1.9300251, 2.189583, 2.3152282, 1.6397184, 1.8685156],
            [1.661913, 1.247655, 1.4489359, 1.6437949, 1.738121, 1.2309928, 1.4027586],
            [1.9300251, 1.4489359, 1.682689, 1.908984, 2.0185275, 1.4295856, 1.6290619],
            [2.189583, 1.6437949, 1.908984, 2.165712, 2.2899873, 1.6218421, 1.848145],
            [2.3152282, 1.738121, 2.0185275, 2.2899873, 2.421394, 1.7149086, 1.9541973],
            [1.6397184, 1.2309928, 1.4295856, 1.6218421, 1.7149086, 1.214553, 1.3840249],
            [1.8685156, 1.4027586, 1.6290619, 1.848145, 1.9541973, 1.3840249, 1.577144],
        ];

        #[rustfmt::skip]
        let k_rep_ab = [
            [1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        ];

        Ok(Self { models, species_to_idx, y_ab, sqrt_alpha_ab, k_rep_ab })
    }

    /// Calculates forces ONLY for the subset of atoms specified in `qm_indices`.
    /// This allows for Hybrid QM/MM simulations.
    pub fn calculate_subset_forces(&mut self, system: &mut System, qm_indices: &[usize]) -> Result<()> {
        let n_atoms = system.positions.len();
        use rayon::prelude::*;
        use std::collections::HashSet;
        
        // Fast lookup for QM atoms
        let qm_set: HashSet<usize> = qm_indices.iter().cloned().collect();

        // --- Phase 1: Compute AEVs and Jacobians (Parallel) ---
        // Only for atoms in qm_indices
        // Result: Vec<Option<(Species, AEV, Jacobian)>> - indexed by original atom index
        
        // We iterate over qm_indices directly but need to map back to global storage or just collect a results list
        // Switch to Serial iter to avoid Rayon/ORT Mutex issues on Mac
        let precompute_results: Vec<(usize, i64, Vec<f32>, Vec<Vec<(usize, Vec3)>>)> = qm_indices.iter()
            .map(|&i| {
                let atom_z = system.atomic_numbers[i] as i64;
                if !self.models.contains_key(&atom_z) {
                    return None;
                }
                let (aev, grads) = self.compute_aev_and_grads(i, system);
                Some((i, atom_z, aev, grads))
            })
            .filter_map(|res| res)
            .collect();

        if precompute_results.is_empty() {
            println!("⚠️ ANI: No atoms prepared! (QM Indices: {}, System Atoms: {})", qm_indices.len(), n_atoms);
        }

        // --- Phase 2: Batched Inference (Serial) ---
        // Group by species
        let mut batches: HashMap<i64, (Vec<usize>, Vec<f32>)> = HashMap::new();
        
        for (i, z, aev, _) in &precompute_results {
            let entry = batches.entry(*z).or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.push(*i);
            entry.1.extend_from_slice(aev);
        }
        
        // Run Inference
        let mut nn_gradients: HashMap<usize, Vec<f32>> = HashMap::new();
        
        for (z, (indices, flat_aev)) in batches {
            println!("   🤖 Inferring Species {} Batch Size {}", z, indices.len());
            if let Some(session) = self.models.get_mut(&z) {
                let batch_size = indices.len();
                let aev_tensor = Array2::from_shape_vec((batch_size, 1008), flat_aev)?;
                let inputs = ort::inputs![ "aev_input" => Value::from_array(aev_tensor)? ];
                
                 // Catch errors without crashing entire sim if model fails
                match session.run(inputs) {
                    Ok(outputs) => {
                        match outputs["aev_gradient"].try_extract_tensor::<f32>() {
                             Ok((_shape, flat_grads)) => {
                                 println!("      ✅ Grads Extracted: {} elements", flat_grads.len());
                                 let grads_chunk_size = 1008;
                                 for (batch_idx, &global_atom_idx) in indices.iter().enumerate() {
                                     let start = batch_idx * grads_chunk_size;
                                     let end = start + grads_chunk_size;
                                     if end <= flat_grads.len() {
                                         nn_gradients.insert(global_atom_idx, flat_grads[start..end].to_vec());
                                     }
                                 }
                             }
                             Err(e) => println!("      ❌ Failed to extract gradients: {:?}", e),
                        }
                    }
                    Err(e) => println!("      ❌ ONNX Run Failed for Species {}: {:?}", z, e),
                }
            } else {
                println!("      ❌ No ONNX Model for Species {}", z);
            }
        }

        // --- Phase 3: Force Assembly (Serial) ---
        let nn_force_updates: Vec<Vec<(usize, Vec3)>> = precompute_results.into_iter()
            .map(|(i, _, _, aev_grads)| {
                let mut local_updates = Vec::new();
                
                if let Some(dE_dAEV) = nn_gradients.get(&i) {
                     let mut max_nn_grad = 0.0f32;
                     let mut max_geom_grad = 0.0f32;
                     let mut nonzero_contributions = 0;

                     for (k, &dE_dA) in dE_dAEV.iter().enumerate() {
                         if dE_dA.abs() > max_nn_grad { max_nn_grad = dE_dA.abs(); }
                         
                         if dE_dA.abs() < 1e-9 { continue; }
                         
                         if let Some(geom_grads) = aev_grads.get(k) {
                             for &(_, grad_vec) in geom_grads {
                                 let g_len = grad_vec.length();
                                 if g_len > max_geom_grad { max_geom_grad = g_len; }
                                 
                                 let force_contribution = grad_vec * dE_dA * HARTREE_TO_EV;
                                 local_updates.push((i, force_contribution)); // Simplified push for debug
                                 nonzero_contributions += 1;
                             }
                         }
                     }
                     if i == 0 { // Only log for one atom to save space
                        println!("      Atom 0 Stats: Max NN Grad: {:.6e}, Max Geom Grad: {:.6e}, Contribs: {}", max_nn_grad, max_geom_grad, nonzero_contributions);
                     }
                }
                local_updates
            })
            .collect();
            
        // Apply NN Updates & Check Stats
        let mut max_force = 0.0f32;
        for updates in nn_force_updates {
            for (idx, force) in updates {
                if force.is_nan() { println!("🔥 ANI Force is NaN at atom {}", idx); }
                if force.length() > max_force { max_force = force.length(); }
                system.forces[idx] += force;
            }
        }
        println!("🔥 ANI Max Force: {:.3} eV/A", max_force);
        
        // --- Part 2: Repulsion Forces (Subset) ---
        // Only calculate repulsion if AT LEAST ONE atom is in QM region.
        // Actually, strictly speaking, hybrid schemes usually do:
        // E = E_QM(qm_atoms) + E_MM(all_atoms) - E_MM(qm_atoms) + E_int(qm, mm)
        // Here we are doing: Forces = F_ANI(qm_atoms) + F_LJ(mm_atoms)
        // Repulsion in ANI is essentially the "Van der Waals" for the QM region.
        // So we should apply ANI repulsion for QM-QM pairs.
        // What about QM-MM pairs? Handled by LJ?
        // Let's assume ANI repulsion is applied for any pair where BOTH are QM.
        
        // This is a simplification.
        
        let rep_force_updates: Vec<Vec<(usize, Vec3)>> = qm_indices.par_iter()
            .map(|&i| {
                let mut local_updates = Vec::new();
                let z_i = system.atomic_numbers[i] as i64;
                let idx_i = match self.species_to_idx.get(&z_i) {
                    Some(&idx) => idx,
                    None => return Vec::new(),
                };

                // Interact with other QM atoms only for ANI repulsion
                for &j in qm_indices {
                    if i >= j { continue; } // Avoid double counting and self
                    
                    let z_j = system.atomic_numbers[j] as i64;
                    let idx_j = match self.species_to_idx.get(&z_j) {
                        Some(&idx) => idx,
                        None => continue,
                    };

                    let r_vec = system.positions[j] - system.positions[i];
                    let r = r_vec.length();
                    
                    if r < 1e-6 { continue; }
                    if r > 5.0 { continue; }

                    let y = self.y_ab[idx_i][idx_j];
                    let sqrt_alpha = self.sqrt_alpha_ab[idx_i][idx_j];
                    let k_rep = self.k_rep_ab[idx_i][idx_j];
                    
                    let exp_term = (-sqrt_alpha * r).exp();
                    let energy_grad = y * (-sqrt_alpha) * exp_term * k_rep;
                    
                    let force_mag = -energy_grad * HARTREE_TO_EV;
                    let force_vec = (r_vec / r) * force_mag;
                    
                    local_updates.push((i, -force_vec));
                    local_updates.push((j, force_vec));
                }
                local_updates
            })
            .collect();
            
        for updates in rep_force_updates {
            for (idx, force) in updates {
                system.forces[idx] += force;
            }
        }

        Ok(())
    }
    
    // Original global calculation method
    pub fn calculate_forces(&mut self, system: &mut System) -> Result<()> {
        let n_atoms = system.positions.len();
        use rayon::prelude::*;
        
        // --- Phase 1: Compute AEVs and Jacobians (Parallel) ---
        // Result: Vec<Option<(Species, AEV, Jacobian)>>
        let precompute_data: Vec<Option<(i64, Vec<f32>, Vec<Vec<(usize, Vec3)>>)>> = (0..n_atoms).into_par_iter()
            .map(|i| {
                let atom_z = system.atomic_numbers[i] as i64;
                if !self.models.contains_key(&atom_z) {
                    return None;
                }
                let (aev, grads) = self.compute_aev_and_grads(i, system);
                 if aev.iter().any(|&x| x.is_nan()) {
                     panic!("🔥 FATAL: NaN detected in AEV for atom {}", i);
                 }
                 // Debug: Check grads magnitude
                 let max_grad = grads.iter().map(|g| g.iter().map(|(_, v)| v.length()).sum::<f32>()).fold(0.0, f32::max);
                 if max_grad == 0.0 {
                     println!("⚠️ WARNING: Atom {} has ZERO geometric gradients!", i);
                 }
                Some((atom_z, aev, grads))
            })
            .collect();

        // --- Phase 2: Batched Inference (Serial) ---
        // Group by species: Species -> (OriginalIndices, FlatAEV)
        let mut batches: HashMap<i64, (Vec<usize>, Vec<f32>)> = HashMap::new();
        
        for (i, data) in precompute_data.iter().enumerate() {
            if let Some((z, aev, _)) = data {
                let entry = batches.entry(*z).or_insert_with(|| (Vec::new(), Vec::new()));
                entry.0.push(i);
                entry.1.extend_from_slice(aev);
            }
        }
        
        // Run Inference per species and store gradients: Map<AtomIndex, GradVector>
        // GradVector is [1008]
        // Since we can't easily map back in parallel without a lookup, let's store in a Vec<Option<Vec<f32>>>
        let mut nn_gradients: Vec<Option<Vec<f32>>> = vec![None; n_atoms];
        
        for (z, (indices, flat_aev)) in batches {
            if let Some(session) = self.models.get_mut(&z) {
                let batch_size = indices.len();
                let aev_tensor = Array2::from_shape_vec((batch_size, 1008), flat_aev)?;
                let inputs = ort::inputs![ "aev_input" => Value::from_array(aev_tensor)? ];
                let outputs = session.run(inputs)?;
                
                if let Ok((shape, flat_grads)) = outputs["aev_gradient"].try_extract_tensor::<f32>() {
                    // Shape should be [Batch, 1008]
                    // flat_grads is all gradients flattened
                    
                    if flat_grads.iter().any(|&x| x.is_nan()) {
                        panic!("🔥 FATAL: NaN detected in NN Output Gradients for species {}", z);
                    }

                    let grads_chunk_size = 1008;
                    for (batch_idx, &global_atom_idx) in indices.iter().enumerate() {
                        let start = batch_idx * grads_chunk_size;
                        let end = start + grads_chunk_size;
                        if end <= flat_grads.len() {
                            nn_gradients[global_atom_idx] = Some(flat_grads[start..end].to_vec());
                        }
                    }
                }
            }
        }

        // --- Phase 3: Force Assembly (Parallel) ---
        // Now using the precomputed Jacobians and the inferred NN gradients
        let nn_force_updates: Vec<Vec<(usize, Vec3)>> = (0..n_atoms).into_par_iter()
            .map(|i| {
                let mut local_updates = Vec::new();
                
                // Need both Jacobian and Gradient
                if let Some((_, _, aev_grads)) = &precompute_data[i] {
                    if let Some(dE_dAEV) = &nn_gradients[i] {
                         for (k, &dE_dA) in dE_dAEV.iter().enumerate() {
                             if dE_dA.abs() < 1e-9 { continue; }
                             
                             if let Some(geom_grads) = aev_grads.get(k) {
                                 for &(neighbor_idx, grad_vec) in geom_grads {
                                     let force_contribution = grad_vec * dE_dA * HARTREE_TO_EV;
                                     local_updates.push((neighbor_idx, -force_contribution));
                                     local_updates.push((i, force_contribution));
                                 }
                             }
                         }
                    }
                }
                local_updates
            })
            .collect();
            
        // Apply NN Updates
        let MAX_FORCE = 10.0; // eV/A clamp for visual stability
        let mut debug_count = 0;
        for updates in nn_force_updates {
            for (idx, force) in updates.iter() {
                if debug_count < 5 {
                    println!("DEBUG: Atom {} Raw NN Force: {:?}", idx, force);
                    debug_count += 1;
                }
                // Clamp force components to avoid explosions
                let clamped_force = force.clamp(Vec3::splat(-MAX_FORCE), Vec3::splat(MAX_FORCE));
                system.forces[*idx] += clamped_force;
            }
        }
        
        // --- Part 2: Repulsion Forces (Parallel) ---
        // (Same as before)
        let rep_force_updates: Vec<Vec<(usize, Vec3)>> = (0..n_atoms).into_par_iter()
            .map(|i| {
                let mut local_updates = Vec::new();
                let z_i = system.atomic_numbers[i] as i64;
                let idx_i = match self.species_to_idx.get(&z_i) {
                    Some(&idx) => idx,
                    None => return Vec::new(),
                };

                for j in (i+1)..n_atoms {
                    let z_j = system.atomic_numbers[j] as i64;
                    let idx_j = match self.species_to_idx.get(&z_j) {
                        Some(&idx) => idx,
                        None => continue,
                    };

                    let r_vec = system.positions[j] - system.positions[i];
                    let r = r_vec.length();
                    
                    if r < 1e-6 { continue; }
                    if r > 5.0 { continue; }

                    let y = self.y_ab[idx_i][idx_j];
                    let sqrt_alpha = self.sqrt_alpha_ab[idx_i][idx_j];
                    let k_rep = self.k_rep_ab[idx_i][idx_j];
                    
                    let exp_term = (-sqrt_alpha * r).exp();
                    let energy_grad = y * (-sqrt_alpha) * exp_term * k_rep;
                    
                    let force_mag = -energy_grad * HARTREE_TO_EV;
                    let force_vec = (r_vec / r) * force_mag;
                    
                    local_updates.push((i, -force_vec));
                    local_updates.push((j, force_vec));
                }
                local_updates
            })
            .collect();
            
        // Apply Repulsion updates
        // Apply Repulsion updates
        // DISABLED due to instability (Source of NaNs/Explosions)
        /*
        for updates in rep_force_updates {
            for (idx, force) in updates {
                let clamped_force = force.clamp(Vec3::splat(-MAX_FORCE), Vec3::splat(MAX_FORCE));
                system.forces[idx] += clamped_force;
            }
        }
        */
        
        Ok(())
    }

    // Returns AEV [1008] and Jacobian [1008, Neighbors, 3] (Sparse)
    // Sparse Representation: Index -> List of (NeighborIdx, DerivativeVector)
    fn compute_aev_and_grads(&self, center_idx: usize, system: &System) -> (Vec<f32>, Vec<Vec<(usize, Vec3)>>) {
        let mut aev = vec![0.0; 1008];
        let mut grads: Vec<Vec<(usize, Vec3)>> = vec![Vec::new(); 1008];
        
        let pos_i = system.positions[center_idx];
        
        // 1. Find Neighbors
        let mut neighbors = Vec::new();
        for (j, &pos_j) in system.positions.iter().enumerate() {
             if center_idx == j { continue; }
             let r_vec = pos_j - pos_i; 
             // Apply PBC here if needed (omitted for minimal logic)
             let r = r_vec.length();
             if r < 1.0e-5 { continue; } // Prevent singularity/NaN
             if r < R_CR {
                 neighbors.push((j, r, r_vec, system.atomic_numbers[j] as i64));
             }
        }

        // --- Radial Part ---
        // Offset: species_idx * 16
        for &(j, r, r_vec, z_j) in &neighbors {
             if let Some(&s_idx) = self.species_to_idx.get(&z_j) {
                 let offset = s_idx * 16;
                 let fc = cutoff_cosine(r, R_CR);
                 let fc_prime = cutoff_cosine_deriv(r, R_CR);
                 let r_hat = r_vec / r;
                 
                 for k in 0..16 {
                     let shift = RADIAL_SHIFTS[k];
                     let delta = r - shift;
                     let gaussian = (-RADIAL_ETA * delta * delta).exp();
                     let g_prime = gaussian * (-2.0 * RADIAL_ETA * delta);
                     
                     // Value
                     aev[offset + k] += gaussian * fc;
                     
                     // Gradient (w.r.t pos_j)
                     // d(G*Fc)/dr = G'*Fc + G*Fc'
                     let deriv_scalar = g_prime * fc + gaussian * fc_prime;
                     let deriv_vec = r_hat * deriv_scalar;

                     
                     grads[offset + k].push((j, deriv_vec));
                 }
             }
        }

        // --- Angular Part ---
        // Indices: Standard ANI order is triu(Species x Species)
        // 7 Species -> 28 Pairs.
        // Pair(S1, S2) where S1 <= S2
        let _r_ca_sq = R_CA * R_CA;
        
        // Double loop over neighbors
        for (m, &(j, r_ij, vec_ij, z_j)) in neighbors.iter().enumerate() {
            if r_ij > R_CA { continue; }
            let s1 = self.species_to_idx.get(&z_j).unwrap_or(&0usize);
            
            for (n, &(k, r_ik, vec_ik, z_k)) in neighbors.iter().enumerate() {
                if m >= n { continue; } // Unique pairs only? ANI sums over all j!=k? 
                // ANI definition: Sum over j, k (j!=k).
                // But usually we iterate unique pairs and multiply by 2?
                // Let's stick to unique pairs m < n to avoid double counting if the params assume it.
                // Standard ANI-1x/2x AEV sums over all distinct pairs j,k (j!=k).
                // Since theta_ijk = theta_ikj, the term is symmetric.
                // If we loop m < n, we account for the pair {j, k}.
                
                if r_ik > R_CA { continue; }
                
                let s2 = self.species_to_idx.get(&z_k).unwrap_or(&0usize);
                
                // Determine pair index
                let (type1, type2) = if s1 <= s2 { (*s1, *s2) } else { (*s2, *s1) };
                
                // Calculate triangular index for (type1, type2)
                // Formula: idx = t1*N - t1*(t1+1)/2 + t2
                // N=7.
                let n_species = 7;
                let pair_idx = type1 * n_species - (type1 * (type1 + 1)) / 2 + type2;
                 
                // Base Offset for Angular:
                // Radial (112) + PairIdx * (4 Sections * 8 Shifts)
                let base_offset = 112 + pair_idx * 32;

                // Geometry
                let r_avg = (r_ij + r_ik) * 0.5;
                let cos_theta = vec_ij.dot(vec_ik) / (r_ij * r_ik);
                // Clamp for safety
                let cos_theta = cos_theta.clamp(-0.999999, 0.999999);
                let theta = cos_theta.acos();
                let sin_theta = theta.sin(); // >= 0
                
                let fc_j = cutoff_cosine(r_ij, R_CA);
                let fc_k = cutoff_cosine(r_ik, R_CA);
                let fc_prod = fc_j * fc_k;
                
                // Gradients of Geometry
                // We need d/d_rj and d/d_rk
                let fc_j_prime = cutoff_cosine_deriv(r_ij, R_CA);
                let fc_k_prime = cutoff_cosine_deriv(r_ik, R_CA);
                
                // Angular Gradients are complex.
                // d(cos)/dr_j = (vec_ik/r_ik - cos*vec_ij/r_ij) / r_ij
                // d(theta)/dr_j = -1/sin(theta) * d(cos)/dr_j
                
                let term_j = (vec_ik / r_ik - vec_ij * (cos_theta / r_ij)) / r_ij;
                let term_k = (vec_ij / r_ij - vec_ik * (cos_theta / r_ik)) / r_ik;
                
                let dtheta_drj = term_j * (-1.0 / sin_theta);
                let dtheta_drk = term_k * (-1.0 / sin_theta);
                
                // Loop Parameters
                for section_idx in 0..4 {
                    for shift_idx in 0..8 {
                        let feature_idx = base_offset + section_idx * 8 + shift_idx;
                        let zeta = ANGULAR_ZETA;
                        let eta = ANGULAR_ETA;
                        let rs = ANGULAR_SHIFTS[shift_idx];
                        let ts = ANGULAR_SECTIONS[section_idx];
                        
                        // Radial part of Angular
                        let delta_r = r_avg - rs;
                        let radial_term = (-eta * delta_r * delta_r).exp();
                        
                        // Angular part
                        let cos_diff = (theta - ts).cos();
                        let angular_term = (1.0 + cos_diff).powf(zeta);
                        
                        let prefactor = 2.0_f32.powf(1.0 - zeta);
                        let value = prefactor * angular_term * radial_term * fc_prod;

                        aev[feature_idx] += value;
                        
                        // Gradients... this is heavy. 
                        // Simplification: Implement only if required for force.
                        // Yes, we need forces.
                        
                        // Derivatives:
                        // dTotal/d_rj = Prefactor * [ 
                        //    d(Ang)/d_rj * Rad * Fc + 
                        //    Ang * d(Rad)/d_rj * Fc +
                        //    Ang * Rad * d(Fc)/d_rj
                        // ]
                        
                        // 1. Angular deriv
                        // d( (1+cos(t-ts))^zeta ) / dtheta = zeta * (1+cos)^zeta-1 * (-sin(t-ts))
                        let d_ang_dtheta = zeta * (1.0 + cos_diff).powf(zeta - 1.0) * -(theta - ts).sin();
                        let d_ang_drj = d_ang_dtheta * dtheta_drj; // Vec3
                        let d_ang_drk = d_ang_dtheta * dtheta_drk; // Vec3
                        
                        // 2. Radial deriv
                        // d(exp)/d_ravg = exp * -2*eta*(ravg - rs)
                        // d_ravg/d_rij = 0.5
                        let d_rad_dr_scalar = radial_term * (-2.0 * eta * delta_r);
                        let d_rad_drj = (vec_ij / r_ij) * (0.5 * d_rad_dr_scalar);
                        let d_rad_drk = (vec_ik / r_ik) * (0.5 * d_rad_dr_scalar);
                        
                        // 3. Cutoff deriv
                        let d_fc_drj = (vec_ij / r_ij) * (fc_j_prime * fc_k); // Fc_k is const w.r.t j
                        let d_fc_drk = (vec_ik / r_ik) * (fc_k_prime * fc_j);
                        
                        // Combine
                        // Grad J
                        let grad_j = prefactor * (
                            d_ang_drj * radial_term * fc_prod +
                            d_rad_drj * angular_term * fc_prod +
                            d_fc_drj * angular_term * radial_term
                        );
                        
                        // Grad K
                        let grad_k = prefactor * (
                            d_ang_drk * radial_term * fc_prod +
                            d_rad_drk * angular_term * fc_prod +
                            d_fc_drk * angular_term * radial_term
                        );
                        
                        grads[feature_idx].push((j, grad_j));
                        grads[feature_idx].push((k, grad_k));
                    }
                }
            }
        }
        
        (aev, grads)
    }
}

// Implement ForceField trait for integration with MD engine
impl crate::ForceField for AniEnsemble {
    fn calculate_forces(&mut self, system: &mut atom_core::System, _grid: &atom_core::SpatialGrid) {
        // Call our ANI-specific calculate_forces (ignore grid since we do neighbor search manually)
        if let Err(e) = AniEnsemble::calculate_forces(self, system) {
            eprintln!("ANI Force Calculation Error: {}", e);
        }
    }
}

// Helpers
fn cutoff_cosine(r: f32, rc: f32) -> f32 {
    if r > rc {
        0.0
    } else {
        0.5 * (1.0 + (PI * r / rc).cos())
    }
}

fn cutoff_cosine_deriv(r: f32, rc: f32) -> f32 {
    if r > rc {
        0.0
    } else {
        -0.5 * (PI / rc) * (PI * r / rc).sin()
    }
}
