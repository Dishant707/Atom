use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;


pub struct Potential {
    session: Session,
    has_forces: bool,
}

impl Potential {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        // Check if the model has a "forces" output
        let has_forces = session.outputs.iter().any(|o| o.name == "forces");
        
        if has_forces {
            println!("🚀 Analytical Forces Detected! Running in Turbo Mode.");
        } else {
            println!("⚠️ No 'forces' output found. Using Finite Difference (Slow).");
        }

        Ok(Self { session, has_forces })
    }

    /// Computes energy and forces for a single molecule.
    /// 
    /// # Arguments
    /// * `species` - Atomic numbers as integers (e.g., 1 for H, 6 for C). Shape: [N_atoms]
    /// * `coordinates` - Atomic positions in Angstroms. Shape: [N_atoms, 3]
    /// 
    /// # Returns
    /// (Energy (Hartree), Forces (Hartree/A flat vector))
    pub fn compute(&mut self, species: &[i64], coordinates: &[f32]) -> Result<(f32, Vec<f32>)> {
        let n_atoms = species.len();

        // --- FAST PATH: Analytical Forces ---
        if self.has_forces {
             // Shape: [1, N]
             let species_array = Array2::from_shape_vec((1, n_atoms), species.to_vec())?;
             // Shape: [1, N, 3]
             let coords_array = Array3::from_shape_vec((1, n_atoms, 3), coordinates.to_vec())?;
             
             let inputs = ort::inputs![
                 "species" => Value::from_array(species_array)?,
                 "coordinates" => Value::from_array(coords_array)?
             ];
             
             // Run Model
             let outputs = self.session.run(inputs)?;
             
             // Validate Forces Output
             if let Ok((shape, forces_data)) = outputs["forces"].try_extract_tensor::<f32>() {
                 // Check if last dimension is 3 (x, y, z)
                 // Shape should be [1, N, 3]
                 if shape.last() == Some(&3) {
                     let energy_val = outputs["energy"].try_extract_tensor::<f32>()?.1[0];
                     return Ok((energy_val, forces_data.to_vec()));
                 } else {
                     eprintln!("⚠️ Warning: 'forces' output has unexpected shape: {:?}. Expected [..., 3]. Falling back to Finite Difference.", shape);
                     // Fallthrough to slow path
                 }
             } else {
                 eprintln!("⚠️ Warning: Could not extract 'forces' tensor. Falling back.");
                 // Fallthrough
             }
         }

        // --- SLOW PATH: Finite Difference ---
        // Fallback for models without gradients baked in
        let delta = 0.001; // Step size for finite difference

        // 1. Prepare Batch Size
        // Batch 0: Original
        // Batches 1..3N: +Delta
        // Batches 3N+1..6N: -Delta
        // Total = 1 + 2 * 3 * N
        let n_dof = n_atoms * 3;
        let batch_size = 1 + 2 * n_dof;

        // 2. Expand Species: Shape [Batch, N]
        let species_vec = species.to_vec();
        let mut species_batch_vec = Vec::with_capacity(batch_size * n_atoms);
        for _ in 0..batch_size {
            species_batch_vec.extend_from_slice(&species_vec);
        }
        let species_array = Array2::from_shape_vec((batch_size, n_atoms), species_batch_vec)?;

        // 3. Expand Coordinates: Shape [Batch, N, 3]
        let coords_vec = coordinates.to_vec();
        let mut coords_batch_vec = Vec::with_capacity(batch_size * n_atoms * 3);
        
        // Batch 0: Original
        coords_batch_vec.extend_from_slice(&coords_vec);

        // Create base Array for easy mutation
        let _base_coords = Array2::from_shape_vec((n_atoms, 3), coords_vec.clone())?;
        
        // Create one giant buffer [BatchSize * N * 3]
        let flat_coords = coordinates; // Already a slice
        let n_floats = flat_coords.len(); // N * 3
        
        // 1. Allocate & Fill: Replicate original molecule 'batch_size' times
        // This is a fast memory copy (memcpy), not a loop of clones
        let mut coords_batch_vec: Vec<f32> = flat_coords.iter().cycle().take(n_floats * batch_size).cloned().collect();

        // 2. Perturb in-place (No allocation inside loop)
        // Batch 0 is Original (Skip)
        
        // Batches 1..=n_dof: +Delta
        for k in 0..n_dof {
            let batch_idx = k + 1;
            let offset = batch_idx * n_floats + k; // Jump to this batch, then to the k-th coordinate
            coords_batch_vec[offset] += delta;
        }

        // Batches (n_dof+1)..=(2*n_dof): -Delta
        for k in 0..n_dof {
            let batch_idx = k + 1 + n_dof;
            let offset = batch_idx * n_floats + k;
            coords_batch_vec[offset] -= delta;
        }

        let coords_array = Array3::from_shape_vec((batch_size, n_atoms, 3), coords_batch_vec)?;

        // 4. Convert to ONNX Runtime Values
        let species_val = Value::from_array(species_array)?;
        let coords_val = Value::from_array(coords_array)?;

        // 5. Run Inference
        let inputs = ort::inputs![
            "species" => species_val,
            "coordinates" => coords_val
        ];

        let outputs = self.session.run(inputs)?;

        // 6. Extract Energies
        let (_, energy_data) = outputs["energy"].try_extract_tensor::<f32>()?;
        let energies: Vec<f32> = energy_data.to_vec();

        if energies.len() != batch_size {
            anyhow::bail!("Unexpected energy output size: expected {}, got {}", batch_size, energies.len());
        }

        let e_original = energies[0];
        
        // 7. Calculate Forces
        let mut forces = Vec::with_capacity(n_dof);
        
        for k in 0..n_dof {
            let idx_plus = 1 + k;
            let idx_minus = 1 + n_dof + k;
            
            let e_plus = energies[idx_plus];
            let e_minus = energies[idx_minus];
            
            let gradient = (e_plus - e_minus) / (2.0 * delta);
            let force = -gradient; // F = -dU/dx
            forces.push(force);
        }

        Ok((e_original, forces))
    }
}
