use crate::ForceField;
use atom_core::{System, SpatialGrid};
use glam::Vec3;
use std::collections::HashMap;
use tch::{CModule, Tensor, Kind, Device, IValue};

pub struct MacePotential {
    model: CModule,
    device: Device,
    cutoff: f32,
}

impl MacePotential {
    pub fn new(model_path: &str, cutoff: f32) -> anyhow::Result<Self> {
        let device = Device::Cpu; // Start with CPU, move to CUDA if available later
        let mut model = CModule::load(model_path)?;
        model.set_eval();
        
        Ok(Self {
            model,
            device,
            cutoff,
        })
    }

    fn compute_graph(&self, system: &System, grid: &SpatialGrid) -> (Tensor, Tensor, Tensor, Tensor) {
        // Build Edge Index and Shifts for MACE
        // Returns: (edge_index, shifts, unit_shifts, cell)
        
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        let mut shifts = Vec::new();
        let mut unit_shifts = Vec::new();
        
        let cutoff_sq = self.cutoff * self.cutoff;
        let box_size = system.box_size;
        
        // MACE usually requires double-sided edges (i->j AND j->i)
        // SpatialGrid usually iterates easy pairs. We need to follow MACE convention.
        
        // Naive O(N^2) or Grid?
        // Let's use Grid for O(N)
        let (dim_x, dim_y, dim_z) = grid.dimensions;
        
        for i in 0..system.positions.len() {
             let pos_i = system.positions[i];
             let cx = (pos_i.x / grid.cell_size).floor() as i32;
             let cy = (pos_i.y / grid.cell_size).floor() as i32;
             let cz = (pos_i.z / grid.cell_size).floor() as i32;
             
             for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                         let nx = cx + dx;
                         let ny = cy + dy;
                         let nz = cz + dz;
                         
                         // PBC Wrap for neighbors
                         // But SpatialGrid cells lookup needs specific index
                         // Assuming SpatialGrid handles naive loop or I check bounds
                         if nx >= 0 && nx < dim_x as i32 && ny >= 0 && ny < dim_y as i32 && nz >= 0 && nz < dim_z as i32 {
                             let cell_idx = (nx as usize) + (ny as usize) * dim_x + (nz as usize) * dim_x * dim_y;
                             if let Some(neighbors) = grid.cells.get(cell_idx) {
                                 for &j_u32 in neighbors {
                                     let j = j_u32 as usize;
                                     if i == j { continue; }
                                     
                                     let pos_j = system.positions[j];
                                     let mut diff = pos_j - pos_i;
                                     let mut shift = Vec3::ZERO;
                                     
                                     // MIC Correction & Shift Calculation
                                     if diff.x > box_size.x * 0.5 { diff.x -= box_size.x; shift.x = -1.0; }
                                     else if diff.x < -box_size.x * 0.5 { diff.x += box_size.x; shift.x = 1.0; }
                                     
                                     if diff.y > box_size.y * 0.5 { diff.y -= box_size.y; shift.y = -1.0; }
                                     else if diff.y < -box_size.y * 0.5 { diff.y += box_size.y; shift.y = 1.0; }
                                     
                                     if diff.z > box_size.z * 0.5 { diff.z -= box_size.z; shift.z = -1.0; }
                                     else if diff.z < -box_size.z * 0.5 { diff.z += box_size.z; shift.z = 1.0; }
                                     
                                     if diff.length_squared() < cutoff_sq {
                                         senders.push(i as i64);
                                         receivers.push(j as i64);
                                         shifts.push([shift.x, shift.y, shift.z]);
                                         unit_shifts.push([shift.x, shift.y, shift.z]); // Usually same for orthogonal
                                     }
                                 }
                             }
                         }
                    }
                }
             }
        }
        
        if senders.is_empty() {
            println!("DEBUG: No neighbors found! Check cutoff or grid.");
            let edge_index = Tensor::zeros(&[2, 0], (Kind::Int64, self.device));
            let shifts_t = Tensor::zeros(&[0, 3], (Kind::Float, self.device));
            let unit_shifts_t = Tensor::zeros(&[0, 3], (Kind::Float, self.device));
            
            let cell_t = Tensor::from_slice(&[
                box_size.x, 0.0, 0.0,
                0.0, box_size.y, 0.0,
                0.0, 0.0, box_size.z
            ]).reshape(&[3, 3]).to(self.device).to_kind(Kind::Float);
            
            return (edge_index, shifts_t, unit_shifts_t, cell_t);
        }
        
        // Debug
        // println!("DEBUG: MACE Graph: {} edges", senders.len());

        let edge_index_vecs: Vec<&[i64]> = vec![&senders, &receivers];
        let edge_index = Tensor::from_slice2(&edge_index_vecs).to(self.device); // [2, N_edges]
        
        let shifts_t = Tensor::from_slice2(&shifts.iter().map(|s| s.as_slice()).collect::<Vec<_>>()).to(self.device).to_kind(Kind::Float);
        let unit_shifts_t = Tensor::from_slice2(&unit_shifts.iter().map(|s| s.as_slice()).collect::<Vec<_>>()).to(self.device).to_kind(Kind::Float);
        
        let cell_t = Tensor::from_slice(&[
            box_size.x, 0.0, 0.0,
            0.0, box_size.y, 0.0,
            0.0, 0.0, box_size.z
        ]).reshape(&[3, 3]).to(self.device).to_kind(Kind::Float);
        
        (edge_index, shifts_t, unit_shifts_t, cell_t)
    }
}

impl ForceField for MacePotential {
    fn calculate_forces(&mut self, system: &mut System, grid: &SpatialGrid) {
        let n_atoms = system.positions.len();
        
        // 1. Prepare Inputs
        // Convert u8 to i64 for torch
        let atomic_numbers_i64: Vec<i64> = system.atomic_numbers.iter().map(|&z| z as i64).collect();
        let z_tensor = Tensor::from_slice(&atomic_numbers_i64)
            .to(self.device);
            
        let pos_flat: Vec<f32> = system.positions.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let pos_tensor = Tensor::from_slice(&pos_flat)
            .reshape(&[n_atoms as i64, 3])
            .to_kind(Kind::Float)
            .to(self.device);
            
        let (edge_index, shifts, unit_shifts, cell) = self.compute_graph(system, grid);
        
        // Batch info
        let batch = Tensor::zeros(&[n_atoms as i64], (Kind::Int64, self.device));
        let ptr = Tensor::from_slice(&[0, n_atoms as i64]).to(self.device);
        
        // Standard MACE deployed keys often use 'atomic_numbers'?
        // Let's assume the user will rename keys if needed.
        // We use common keys for 'mace_create_model'.
        
        let mut input_dict = Vec::new();
        // Typically: node_attrs, positions, edge_index, cell, shifts
        // Note: node_attrs usually implies OneHot. 'atomic_numbers' implies Raw Z.
        // We try passing 'node_attrs' as the atomic numbers (some models support this)
        input_dict.push(("node_attrs".to_string(), z_tensor)); 
        input_dict.push(("positions".to_string(), pos_tensor));
        input_dict.push(("edge_index".to_string(), edge_index));
        input_dict.push(("shifts".to_string(), shifts));
        // input_dict.push(("unit_shifts".to_string(), unit_shifts)); // Often redundant but required by some
        input_dict.push(("cell".to_string(), cell));
        input_dict.push(("batch".to_string(), batch));
        input_dict.push(("ptr".to_string(), ptr));
        
        // LAMMPS_MACE signature: forward(data: Dict, local_or_ghost: Tensor, compute_virials: bool)
        
        // 2. local_or_ghost: [N_atoms] (1.0 for local)
        let local_icon = Tensor::ones(&[n_atoms as i64], (Kind::Float, self.device));
        
        // 3. compute_virials: bool
        let compute_virials = IValue::Bool(false);
        
        let dict_ivalue = IValue::GenericDict(
            input_dict.into_iter().map(|(k, v)| (IValue::String(k), IValue::Tensor(v))).collect()
        );
        
        // Arguments: [data, local_or_ghost, compute_virials]
        match self.model.forward_is(&[dict_ivalue, IValue::Tensor(local_icon), compute_virials]) {
            Ok(output) => {
                 println!("DEBUG: MACE Output Structure: {:?}", output);
                 // Output is usually Dict. Get 'forces'
                 if let IValue::GenericDict(out_map) = output {
                     // Find "forces"
                      let mut found_forces = false;
                      for (k, v) in out_map {
                          if let IValue::String(key) = k {
                              if key == "forces" {
                                  if let IValue::Tensor(f_tensor) = v {
                                      // Convert to Float (f32) as MACE output might be Double (f64)
                                      let f_cpu = f_tensor.to_kind(Kind::Float).to_device(Device::Cpu);
                                      
                                      // Use TryFrom
                                      if let Ok(f_vec) = Vec::<f32>::try_from(f_cpu) {
                                          if f_vec.iter().any(|x| x.is_nan()) {
                                              println!("🔥 MACE returned NaN forces!");
                                          }
                                          // Debug first atom
                                          if !f_vec.is_empty() {
                                              println!("DEBUG: MACE Force[0]: [{}, {}, {}]", f_vec[0], f_vec[1], f_vec[2]);
                                          }
                                          
                                          for i in 0..n_atoms {
                                              if i*3+2 < f_vec.len() {
                                                  let fx = f_vec[i*3+0] as f32; // f64 if double precision?
                                                  let fy = f_vec[i*3+1] as f32;
                                                  let fz = f_vec[i*3+2] as f32;
                                                  system.forces[i] += Vec3::new(fx, fy, fz);
                                              }
                                          }
                                          found_forces = true;
                                      } else {
                                          println!("⚠️ Failed to convert forces tensor to Vec<f32>");
                                      }
                                  }
                              }
                          }
                      }
                      if !found_forces {
                          println!("⚠️ MACE Output did not contain 'forces' key");
                      }
                 } else {
                     println!("⚠️ MACE Output was not a Dictionary");
                 }
            },
            Err(e) => {
                println!("🔥 MACE Forward Error: {:?}", e);
            }
        }
    }
}
