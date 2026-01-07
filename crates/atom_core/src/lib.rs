use glam::Vec3;
pub mod dcd;
pub mod potential;
pub mod surface;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use rayon::prelude::*;

// Molecule identification and classification
#[derive(Debug, Clone, PartialEq)]
pub enum MoleculeType {
    Protein,
    DNA,
    RNA,
    SmallMolecule,
    Ligand,
    Complex,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Classification {
    Organic,
    Inorganic,
    Organometallic,
    Unknown,
}

// Secondary structure classification for proteins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
    Unknown,
}

impl Default for BondOrder {
    fn default() -> Self {
        BondOrder::Single
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecondaryStructure {
    Helix,    // α-helix
    Sheet,    // β-sheet
    Loop,     // Loops, turns, coils
    Unknown,
}

impl Default for SecondaryStructure {
    fn default() -> Self {
        SecondaryStructure::Unknown
    }
}


#[derive(Debug, Clone)]
pub struct MoleculeInfo {
    pub name: Option<String>,
    pub formula: Option<String>,
    pub mol_type: MoleculeType,
    pub classification: Classification,
}

impl Default for MoleculeInfo {
    fn default() -> Self {
        Self {
            name: None,
            formula: None,
            mol_type: MoleculeType::Unknown,
            classification: Classification::Unknown,
        }
    }
}

// Kyte-Doolittle Hydrophobicity Scale
// Positive = Hydrophobic (buried), Negative = Hydrophilic (surface)
pub fn get_hydrophobicity(residue: &str) -> Option<f32> {
    match residue.trim().to_uppercase().as_str() {
        "ALA" | "A" => Some(1.8),
        "ARG" | "R" => Some(-4.5),
        "ASN" | "N" => Some(-3.5),
        "ASP" | "D" => Some(-3.5),
        "CYS" | "C" => Some(2.5),
        "GLN" | "Q" => Some(-3.5),
        "GLU" | "E" => Some(-3.5),
        "GLY" | "G" => Some(-0.4),
        "HIS" | "H" => Some(-3.2),
        "ILE" | "I" => Some(4.5),
        "LEU" | "L" => Some(3.8),
        "LYS" | "K" => Some(-3.9),
        "MET" | "M" => Some(1.9),
        "PHE" | "F" => Some(2.8),
        "PRO" | "P" => Some(-1.6),
        "SER" | "S" => Some(-0.8),
        "THR" | "T" => Some(-0.7),
        "TRP" | "W" => Some(-0.9),
        "TYR" | "Y" => Some(-1.3),
        "VAL" | "V" => Some(4.2),
        _ => None,
    }
}


#[derive(Default, Debug, Clone)]
pub struct System {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub masses: Vec<f32>,
    pub ids: Vec<u32>,
    pub atomic_numbers: Vec<u8>,
    pub bonds: Vec<(u32, u32, BondOrder)>, // Pairs of atom indices + Order
    // Simulation box (min, max)
    pub box_size: Vec3,
    // Biological Assembly Transforms (from BIOMT)
    pub assembly_transforms: Vec<glam::Mat4>,
    // Chemical Identification
    pub chain_ids: Vec<String>,
    pub residue_names: Vec<String>,
    pub residue_ids: Vec<u32>,
    // Secondary Structure (for proteins)
    pub secondary_structure: Vec<SecondaryStructure>,
    // Molecule Information
    pub molecule_info: MoleculeInfo,
    // Atom Names (for Backbone Trace)
    pub atom_names: Vec<String>,
}

impl System {
    pub fn new(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            forces: Vec::with_capacity(capacity),
            masses: Vec::with_capacity(capacity),
            ids: Vec::with_capacity(capacity),
            atomic_numbers: Vec::with_capacity(capacity),
            bonds: Vec::new(),
            box_size: Vec3::new(100.0, 100.0, 100.0),
            assembly_transforms: Vec::new(),
            chain_ids: Vec::with_capacity(capacity),
            residue_names: Vec::with_capacity(capacity),
            residue_ids: Vec::with_capacity(capacity),
            secondary_structure: Vec::with_capacity(capacity),
            molecule_info: MoleculeInfo::default(),
            atom_names: Vec::with_capacity(capacity),
        }
    }

    pub fn has_structural_metadata(&self) -> bool {
        self.secondary_structure.iter().any(|&s| s == SecondaryStructure::Helix || s == SecondaryStructure::Sheet)
    }

    pub fn add_atom(&mut self, pos: Vec3, mass: f32, atomic_number: u8, chain_id: String, residue_name: String, residue_id: u32, secondary_structure: SecondaryStructure, atom_name: String) {
        self.positions.push(pos);
        self.velocities.push(Vec3::ZERO);
        self.forces.push(Vec3::ZERO);
        self.masses.push(mass);
        self.ids.push(self.ids.len() as u32);
        self.atomic_numbers.push(atomic_number);
        self.chain_ids.push(chain_id);
        self.residue_names.push(residue_name);
        self.residue_ids.push(residue_id);
        self.secondary_structure.push(secondary_structure);
        self.atom_names.push(atom_name);
    }
    
    pub fn expand_assembly(&mut self) {
        if self.assembly_transforms.is_empty() { return; }
        
        println!("Expanding Biological Assembly with {} matrices...", self.assembly_transforms.len());
        
        let n_src = self.positions.len();
        let n_bonds_src = self.bonds.len();
        let total_atoms = n_src * self.assembly_transforms.len();
        
        let mut new_pos = Vec::with_capacity(total_atoms);
        let mut new_vel = Vec::with_capacity(total_atoms);
        let mut new_force = Vec::with_capacity(total_atoms);
        let mut new_mass = Vec::with_capacity(total_atoms);
        let mut new_ids = Vec::with_capacity(total_atoms);
        let mut new_Z = Vec::with_capacity(total_atoms);
        let mut new_bonds = Vec::with_capacity(n_bonds_src * self.assembly_transforms.len());
        let mut new_chain_ids = Vec::with_capacity(total_atoms);
        let mut new_residue_names = Vec::with_capacity(total_atoms);
        let mut new_residue_ids = Vec::with_capacity(total_atoms);
        let mut new_secondary_structure = Vec::with_capacity(total_atoms);
        let mut new_atom_names = Vec::with_capacity(total_atoms);
        
        let mut min_bound = Vec3::splat(f32::MAX);
        let mut max_bound = Vec3::splat(f32::MIN);
        
        let mut atom_counter = 0;
        
        for mat in &self.assembly_transforms {
             let offset = atom_counter as u32;
             
             // Transform Atoms
             for i in 0..n_src {
                 let p = self.positions[i];
                 // Apply Transform (Mat4 * Vec3 -> treat as POINT w/ w=1)
                 let p4 = *mat * glam::Vec4::new(p.x, p.y, p.z, 1.0);
                 let p_new = Vec3::new(p4.x, p4.y, p4.z);
                 
                 new_pos.push(p_new);
                 new_vel.push(self.velocities[i]); // Keep velocity (or rotate it? Assume static for now)
                 new_force.push(Vec3::ZERO);
                 new_mass.push(self.masses[i]);
                 new_ids.push(atom_counter as u32);
                 new_Z.push(self.atomic_numbers[i]);
                 new_chain_ids.push(self.chain_ids[i].clone());
                 new_residue_names.push(self.residue_names[i].clone());
                 new_residue_ids.push(self.residue_ids[i]);
                 if i < self.secondary_structure.len() {
                     new_secondary_structure.push(self.secondary_structure[i]);
                 } else {
                     new_secondary_structure.push(SecondaryStructure::Unknown);
                 }
                 if i < self.atom_names.len() {
                     new_atom_names.push(self.atom_names[i].clone());
                 } else {
                     new_atom_names.push("UNK".to_string());
                 }
                 
                 min_bound = min_bound.min(p_new);
                 max_bound = max_bound.max(p_new);
                 
                 atom_counter += 1;
             }
             
             // Transform Bonds
             for &(a, b, order) in &self.bonds {
                 new_bonds.push((a + offset, b + offset, order));
             }
        }
        
        // Swap Data
        self.positions = new_pos;
        self.velocities = new_vel;
        self.forces = new_force;
        self.masses = new_mass;
        self.ids = new_ids;
        self.atomic_numbers = new_Z;
        self.bonds = new_bonds;
        self.chain_ids = new_chain_ids;
        self.residue_names = new_residue_names;
        self.residue_ids = new_residue_ids;
        self.secondary_structure = new_secondary_structure;
        self.atom_names = new_atom_names;

        
        // Update Box Size & Center
        let size = max_bound - min_bound;
        self.box_size = size + 20.0; // padding
        let center = (min_bound + max_bound) * 0.5;
        let box_center = self.box_size * 0.5;
        let shift = box_center - center;
        
        for p in &mut self.positions {
            *p += shift;
        }
        
        println!("Assembly Expanded: {} atoms, {} bonds", self.positions.len(), self.bonds.len());
    }
    
    // Auto-detect molecule information from composition
    pub fn auto_detect_molecule_info(&mut self) {
        // Detect molecule type from residues
        let amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"];
        let nucleotides_dna = ["DA", "DT", "DG", "DC", "A", "T", "G", "C"];
        let _nucleotides_rna = ["A", "U", "G", "C"];
        
        let has_amino_acids = self.residue_names.iter().any(|r| amino_acids.contains(&r.as_str()));
        let has_dna = self.residue_names.iter().any(|r| nucleotides_dna.contains(&r.as_str()));
        let has_rna = self.residue_names.iter().any(|r| r == "U");
        
        self.molecule_info.mol_type = if has_amino_acids {
            MoleculeType::Protein
        } else if has_rna {
            MoleculeType::RNA
        } else if has_dna {
            MoleculeType::DNA
        } else if self.positions.len() < 100 {
            MoleculeType::SmallMolecule
        } else {
            MoleculeType::Complex
        };
        
        // Classify as organic/inorganic
        let elements: std::collections::HashSet<u8> = self.atomic_numbers.iter().copied().collect();
        let has_carbon = elements.contains(&6);
        let has_hydrogen = elements.contains(&1);
        let metals = [26, 29, 30, 12, 20, 11, 19, 25, 27, 28]; // Fe, Cu, Zn, Mg, Ca, Na, K, Mn, Co, Ni
        let has_metal = elements.iter().any(|&e| metals.contains(&e));
        
        // Proteins and DNA/RNA are always organic, even if H atoms are missing
        let is_biomolecule = matches!(self.molecule_info.mol_type, 
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA);
        
        self.molecule_info.classification = if is_biomolecule {
            Classification::Organic
        } else if has_carbon && has_hydrogen {
            if has_metal {
                Classification::Organometallic
            } else {
                Classification::Organic
            }
        } else if has_carbon {
            // Has carbon but no hydrogen (could be missing H or truly inorganic)
            if has_metal {
                Classification::Organometallic
            } else {
                Classification::Organic  // Assume organic if has carbon
            }
        } else {
            Classification::Inorganic
        };
        
        // Calculate formula if not set
        if self.molecule_info.formula.is_none() {
            let mut counts: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
            for &z in &self.atomic_numbers {
                *counts.entry(z).or_insert(0) += 1;
            }
            
            // Format as Hill notation (C, H, then alphabetical)
            let mut formula = String::new();
            if let Some(&c_count) = counts.get(&6) {
                formula.push_str(&format!("C{} ", c_count));
            }
            if let Some(&h_count) = counts.get(&1) {
                formula.push_str(&format!("H{} ", h_count));
            }
            for (&z, &count) in counts.iter() {
                if z != 6 && z != 1 {
                    let elem = match z {
                        7 => "N", 8 => "O", 15 => "P", 16 => "S", 26 => "Fe", 12 => "Mg",
                        _ => "X"
                    };
                    formula.push_str(&format!("{}{} ", elem, count));
                }
            }
            self.molecule_info.formula = Some(formula.trim().to_string());
        }
    }
    
    // Simple PDB Parser (subset)
    // ATOM      1  N   ILE A  16      60.609  63.518  42.529  1.00 17.06           N
    pub fn from_pdb<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        
        let mut system = System::new(1000);
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        
        let mut current_biomt = [0.0; 12]; // r11, r12, r13, t1...
        let mut biomt_rows_found = 0;
        
        // Secondary Structure Ranges
        // (ChainID, StartSeq, EndSeq, Type)
        let mut helices: Vec<(String, u32, u32)> = Vec::new();
        let mut sheets: Vec<(String, u32, u32)> = Vec::new();
        let mut ss_map: std::collections::HashMap<(String, u32), SecondaryStructure> = std::collections::HashMap::new();

        // Metadata buffers
        let mut title = String::new();
        let mut classification = Classification::Unknown;
        
        let mut lines_iterator = reader.lines();
        while let Some(line) = lines_iterator.next() {
            let line = line?;
            
            // Parse HEADER for Classification
            // HEADER    PHOTOSYNTHESIS                          21-FEB-25   9LZK 
            if line.starts_with("HEADER") && line.len() > 40 {
                let class_str = line[10..40].trim();
                classification = match class_str {
                    "DNA" | "RNA" | "PROTEIN" => Classification::Organic,
                    _ => Classification::Organic, // Default to organic for PDBs usually
                };
            }

            // Parse TITLE
            // TITLE     THE PSI1-ISIA13 COMPLEX...
            // TITLE    2 MONOMERIC PSI CORE
            if line.starts_with("TITLE ") {
                if line.len() > 10 {
                    let fragment = line[10..].trim();
                    // handle continuation 
                    // (Real PDB parsing is complex, for now just append space)
                     if !title.is_empty() { title.push(' '); }
                     title.push_str(fragment);
                }
            }
            
            // Parse BIOMT
            if line.starts_with("REMARK 350") && line.contains("BIOMT") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 8 {
                    // parts[2]: BIOMT1/2/3
                    // parts[4..8]: values
                    // "REMARK", "350", "BIOMT1", "1", "1.000000", "0.000000", "0.000000", "0.00000"
                    if let (Ok(v1), Ok(v2), Ok(v3), Ok(v4)) = (
                        parts[4].parse::<f32>(), parts[5].parse::<f32>(), parts[6].parse::<f32>(), parts[7].parse::<f32>()
                    ) {
                         if parts[2] == "BIOMT1" {
                             current_biomt[0] = v1; current_biomt[1] = v2; current_biomt[2] = v3; current_biomt[3] = v4;
                             biomt_rows_found |= 1;
                         } else if parts[2] == "BIOMT2" {
                             current_biomt[4] = v1; current_biomt[5] = v2; current_biomt[6] = v3; current_biomt[7] = v4;
                             biomt_rows_found |= 2;
                         } else if parts[2] == "BIOMT3" {
                             current_biomt[8] = v1; current_biomt[9] = v2; current_biomt[10] = v3; current_biomt[11] = v4;
                             biomt_rows_found |= 4;
                         }
                         
                         if biomt_rows_found == 7 { // 1 | 2 | 4
                             // Complete Matrix
                             let mat = glam::Mat4::from_cols_array(&[
                                 current_biomt[0], current_biomt[4], current_biomt[8], 0.0, // Col 1
                                 current_biomt[1], current_biomt[5], current_biomt[9], 0.0, // Col 2
                                 current_biomt[2], current_biomt[6], current_biomt[10], 0.0, // Col 3
                                 current_biomt[3], current_biomt[7], current_biomt[11], 1.0  // Col 4 (Translation)
                             ]);
                             system.assembly_transforms.push(mat);
                             biomt_rows_found = 0; // Reset
                         }
                    }
                }
            }
            
            // Parse HELIX
            // COLUMNS 20 (Chain), 22-25 (InitSeq), 34 (EndChain), 36-39 (EndSeq)
            if line.starts_with("HELIX ") {
               if line.len() >= 40 {
                   let _chain_id = line[19..20].trim().to_string(); // Sometimes it's stuck elsewhere? standard says 20
                   // Actually PDB format: 
                   // 7-10 Serial
                   // 12-14 HelixID
                   // 16-18 ResName
                   // 20    ChainID  (Index 19)
                   // 22-25 SeqNum
                   
                   // Let's use lenient parsing for ChainID if strict fails or is space?
                   // Just trim and ensure it works
                   
                   // Strict
                   let chain_id = line[19..20].to_string();
                   let _init_seq = line[21..25].trim().parse::<u32>().unwrap_or(0);
                   let _end_seq = line[33..37].trim().parse::<u32>().unwrap_or(0); // 34-37 is EndSeq usually?
                   // Wait, original code had 36..40. PDB spec says 34-37?
                   // Col 34-37 is end seqnum.
                   
                   // Let's fix indices based on standard PDB
                   // Init: 22-25 -> Index 21..25
                   // End:  34-37 -> Index 33..37
                   let init_seq = line[21..25].trim().parse::<u32>().unwrap_or(0);
                   let end_seq = line[33..37].trim().parse::<u32>().unwrap_or(0);
                   helices.push((chain_id.clone(), init_seq, end_seq));
                   for i in init_seq..=end_seq {
                       ss_map.insert((chain_id.clone(), i), SecondaryStructure::Helix);
                   }
               }
            }
            
            // Parse SHEET
            // COLUMNS 22 (Chain), 23-26 (InitSeq), 33 (EndChain), 34-37 (EndSeq)
            if line.starts_with("SHEET ") {
               if line.len() >= 38 {
                   let chain_id = line[21..22].to_string();
                   let init_seq = line[22..26].trim().parse::<u32>().unwrap_or(0);
                   let end_seq = line[33..37].trim().parse::<u32>().unwrap_or(0);
                   sheets.push((chain_id.clone(), init_seq, end_seq));
                   for i in init_seq..=end_seq {
                       ss_map.insert((chain_id.clone(), i), SecondaryStructure::Sheet);
                   }
               }
            }
            
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                // Fixed column width parsing standard for PDB
                // x: 30-38, y: 38-46, z: 46-54
                if line.len() < 54 { continue; }
                
                let x_str = &line[30..38].trim();
                let y_str = &line[38..46].trim();
                let z_str = &line[46..54].trim();
                
                // Element: 76-78
                let element_str = if line.len() >= 78 {
                    line[76..78].trim()
                } else {
                    ""
                };

                let atomic_number = match element_str {
                    "H" => 1,
                    "C" => 6,
                    "N" => 7,
                    "O" => 8,
                    "S" => 16,
                    "P" => 15,
                    _ => {
                        // Fallback to inference from Name (12-16)
                         let name = &line[12..16].trim();
                         if name.starts_with("C") { 6 }
                         else if name.starts_with("N") { 7 }
                         else if name.starts_with("O") { 8 }
                         else if name.starts_with("S") { 16 }
                         else if name.starts_with("H") { 1 }
                         else { 6 } // Default C
                    }
                };
                
                let mass = match atomic_number {
                    1 => 1.008,
                    6 => 12.01,
                    7 => 14.01,
                    8 => 16.00,
                    15 => 30.97,
                    16 => 32.06,
                    _ => 12.0,
                };
                
                // Parse metadata
                let chain_id = if line.len() > 21 {
                    line.chars().nth(21).unwrap_or('A').to_string()
                } else {
                    "A".to_string()
                };
                
                let residue_name = if line.len() >= 20 {
                    line[17..20].trim().to_string()
                } else {
                    "UNK".to_string()
                };
                
                let residue_id = if line.len() >= 26 {
                    line[22..26].trim().parse::<u32>().unwrap_or(0)
                } else {
                    0
                };
                
                if let (Ok(x), Ok(y), Ok(z)) = (x_str.parse::<f32>(), y_str.parse::<f32>(), z_str.parse::<f32>()) {
                    let pos = Vec3::new(x, y, z);
                    min = min.min(pos);
                    max = max.max(pos);
                    
                    // Determine Secondary Structure (Optimized O(1) Lookup)
                    let ss = if let Some(&s) = ss_map.get(&(chain_id.clone(), residue_id)) {
                        s
                    } else {
                        SecondaryStructure::Loop
                    };
                    
                    let atom_name_str = if line.len() >= 16 {
                        line[12..16].trim().to_string()
                    } else {
                        "UNK".to_string()
                    };

                    system.add_atom(pos, mass, atomic_number, chain_id, residue_name, residue_id, ss, atom_name_str); 
                }
            }
        }
        
        // Apply Biological Assembly Transforms (if any)
        system.expand_assembly();
        
        // Recalculate bounds from current positions (in case assembly expansion changed them)
        if !system.positions.is_empty() {
             min = system.positions[0];
             max = system.positions[0];
             for &p in &system.positions {
                 min = min.min(p);
                 max = max.max(p);
             }
        }

        // Auto-center box?
        let size = max - min;
        system.box_size = size + 10.0; // padding
        
        // Center atoms in box
        let center = (min + max) * 0.5;
        let box_center = system.box_size * 0.5;
        let offset = box_center - center;
        for pos in &mut system.positions {
           *pos += offset;
        }
        
        // Infer Bonds based on distance
        // 2.2A Safe upper bound for covalent bonds (including S-S at ~2.05A)
        // Optimization: For large systems (>500k atoms), skip bond inference to speed up loading.
        if system.positions.len() <= 500_000 {
            let cutoff_sq = 2.45 * 2.45;
            
            // Initialize Grid for fast lookup
            // Use cell size slightly larger than cutoff (2.5 > 2.2)
            let cell_size = 2.5;
            let mut grid = SpatialGrid::new(system.box_size, cell_size);
            grid.insert(&system);
            
            let (dim_x, dim_y, dim_z) = grid.dimensions;
            
            // Limit checks to prevent hanging on massive grids
            if dim_x * dim_y * dim_z < 100_000_000 {
                // Rayon Parallel Iterator: ~10x speedup
                let new_bonds: Vec<(u32, u32, BondOrder)> = system.positions.par_iter().enumerate()
                    .flat_map(|(i, &pos)| {
                        let mut local_bonds = Vec::new();
                        let cx = (pos.x / cell_size).floor() as i32;
                        let cy = (pos.y / cell_size).floor() as i32;
                        let cz = (pos.z / cell_size).floor() as i32;
                        
                        // Query 3x3x3 neighborhood
                        for dx in -1..=1 {
                            for dy in -1..=1 {
                                for dz in -1..=1 {
                                    let nx = cx + dx;
                                    let ny = cy + dy;
                                    let nz = cz + dz;
                                    
                                    if nx >= 0 && nx < dim_x as i32 &&
                                       ny >= 0 && ny < dim_y as i32 &&
                                       nz >= 0 && nz < dim_z as i32 
                                    {
                                        let cell_idx = (nx as usize) + (ny as usize) * dim_x + (nz as usize) * dim_x * dim_y;
                                        if cell_idx < grid.cells.len() {
                                            for &j in &grid.cells[cell_idx] {
                                                // Only add bond if i < j to avoid duplicates and self-bonds
                                                if (i as u32) < j {
                                                    let diff = pos - system.positions[j as usize];
                                                    if diff.length_squared() < cutoff_sq {
                                                        local_bonds.push((i as u32, j, BondOrder::Single));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        local_bonds
                    })
                    .collect();
                
                system.bonds.extend(new_bonds);

            } else {
                 println!("Warning: Grid too large, skipping bond inference.");
            }
        } else {
            println!("Large system detected ({} atoms). Skipping bond inference for speed.", system.positions.len());
        }
        
        println!("Inferred {} bonds from {} atoms", system.bonds.len(), system.positions.len());
        println!("Found {} Helix ranges and {} Sheet ranges", helices.len(), sheets.len());
        
        println!("Inferred {} bonds from {} atoms", system.bonds.len(), system.positions.len());
        println!("Found {} Helix ranges and {} Sheet ranges", helices.len(), sheets.len());

        // Update Molecule Info
        if !title.is_empty() {
            system.molecule_info.name = Some(title);
        } else {
             system.molecule_info.name = Some(path.as_ref().file_stem().unwrap().to_string_lossy().to_string());
        }
        
        system.molecule_info.classification = classification;

        // Calculate Formula
        let mut elem_counts = std::collections::HashMap::new();
        for &z in &system.atomic_numbers {
            *elem_counts.entry(z).or_insert(0) += 1;
        }
        // Hill System Order: C first, then H, then alphabetical
        let mut formula = String::new();
        if let Some(&c_count) = elem_counts.get(&6) {
            formula.push_str(&format!("C{}", c_count));
            elem_counts.remove(&6);
             if let Some(&h_count) = elem_counts.get(&1) {
                formula.push_str(&format!("H{}", h_count));
                elem_counts.remove(&1);
            }
        }
        let mut sorted_elems: Vec<_> = elem_counts.keys().cloned().collect();
        sorted_elems.sort();
        for z in sorted_elems {
            let sym = match z {
                1 => "H", 6 => "C", 7 => "N", 8 => "O", 15 => "P", 16 => "S", 12 => "Mg", 26 => "Fe", 
                _ => "X" 
            };
            let count = elem_counts[&z];
            formula.push_str(&format!("{}{}", sym, count));
        }
        system.molecule_info.formula = Some(formula);
        
        Ok(system)
    }

    pub fn from_xyz<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Line 1: Atom Count
        let count_line = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty XYZ file"))??;
        let n_atoms = count_line.trim().parse::<usize>().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid atom count"))?;

        let mut system = System::new(n_atoms);
        
        // Line 2: Comment (Skip)
        if let Some(res) = lines.next() { res?; }

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 { continue; }

            let el_str = parts[0];
            let x = parts[1].parse::<f32>().unwrap_or(0.0);
            let y = parts[2].parse::<f32>().unwrap_or(0.0);
            let z = parts[3].parse::<f32>().unwrap_or(0.0);

            let atomic_number = match el_str {
                "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5, "C" => 6, "N" => 7, "O" => 8, "F" => 9, "Ne" => 10,
                "Na" => 11, "Mg" => 12, "Al" => 13, "Si" => 14, "P" => 15, "S" => 16, "Cl" => 17, "Ar" => 18, "K" => 19, "Ca" => 20,
                _ => 6, // Default C
            };
            
            let mass = match atomic_number {
                1 => 1.008, 6 => 12.01, 7 => 14.01, 8 => 16.00, 16 => 32.06, _ => 12.0,
            };

            let pos = Vec3::new(x, y, z);
            min = min.min(pos);
            max = max.max(pos);
            system.add_atom(pos, mass, atomic_number, "A".to_string(), "UNK".to_string(), 0, SecondaryStructure::Unknown, el_str.to_string());
        }

        // Center atoms
        let center = (min + max) * 0.5;
        // Box size inference (XYZ doesn't have it, assume padding)
        let size = max - min;
        system.box_size = size + 20.0;
        let offset = system.box_size * 0.5 - center;
        
        for pos in &mut system.positions {
            *pos += offset;
        }

        // Infer Bonds
        let cutoff_sq = 2.2 * 2.2;
        let cell_size = 2.5;
        let mut grid = SpatialGrid::new(system.box_size, cell_size);
        grid.insert(&system);
        
        let (dim_x, dim_y, dim_z) = grid.dimensions;
        for i in 0..system.positions.len() {
            let pos = system.positions[i];
            let cx = (pos.x / cell_size).floor() as i32;
            let cy = (pos.y / cell_size).floor() as i32;
            let cz = (pos.z / cell_size).floor() as i32;
            
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        let nz = cz + dz;
                        if nx >= 0 && nx < dim_x as i32 && ny >= 0 && ny < dim_y as i32 && nz >= 0 && nz < dim_z as i32 {
                            let cell_idx = (nx as usize) + (ny as usize) * dim_x + (nz as usize) * dim_x * dim_y;
                            if cell_idx < grid.cells.len() {
                                for &j in &grid.cells[cell_idx] {
                                    if (i as u32) < j {
                                        let diff = pos - system.positions[j as usize];
                                        if diff.length_squared() < cutoff_sq {
                                            system.bonds.push((i as u32, j, BondOrder::Single));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("Inferred {} bonds from XYZ", system.bonds.len());
        Ok(system)
    }

    pub fn from_cif<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let _system = System::new(1000);
        let _min = Vec3::splat(f32::MAX);
        let _max = Vec3::splat(f32::MIN);
        
        let _in_atom_loop = false;
        let _col_indices: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let _loop_columns: Vec<String> = Vec::new();
        
        // CIF Secondary Structure Storage
        // (ChainID, StartSeq, EndSeq, Type)
        let helices: Vec<(String, u32, u32)> = Vec::new();
        let sheets: Vec<(String, u32, u32)> = Vec::new();
        
        // Need to track which loop we are in
        let _current_loop_category = String::new();
        
        // Read entire file into memory (required for par_iter)
        // For 100MB file, this is fine.
        let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
        
        // Parallel Parse Structure:
        // 1. Scan for loop headers sequentially to build column maps.
        // 2. Identify the range of lines corresponding to atoms.
        // 3. Parallel parse that range.
        
        let mut atom_lines_start = 0;
        let mut atom_lines_end = 0;
        let mut found_atoms = false;
        
        // Scan headers
        let mut col_indices: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut loop_columns: Vec<String> = Vec::new();
        let mut current_loop_category = String::new();
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
            
            if trimmed == "loop_" {
                loop_columns.clear();
                col_indices.clear();
                current_loop_category.clear();
                if found_atoms { break; } // If we finished an atom block, stop scanning? Or CIF might have multiple? Assuming single model.
                continue;
            }
            
            if trimmed.starts_with("_atom_site.") {
                current_loop_category = "atom_site".to_string();
                loop_columns.push(trimmed.to_string());
                col_indices.insert(trimmed.to_string(), loop_columns.len() - 1);
                continue;
            }
            
            // Check for data start
            if current_loop_category == "atom_site" && !trimmed.starts_with('_') && !trimmed.starts_with("loop_") {
                 // Found start of atoms
                 atom_lines_start = i;
                 // Find end
                 let mut end = i;
                 for j in i..lines.len() {
                     let t = lines[j].trim();
                     if t.starts_with("loop_") || t.starts_with('#') && t.contains("END") { // Heuristic
                         break;
                     }
                     if t.starts_with('_') { break; } // New category
                     end = j;
                 }
                 atom_lines_end = end + 1;
                 found_atoms = true;
                 break; 
            }
        }
        
        let mut system = System::new(1000);
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        
        if found_atoms {
            use rayon::prelude::*;
            
            
            println!("Parallel parsing {} atom lines...", atom_lines_end - atom_lines_start);
            
            // Define a struct to hold parsed atom data to avoid Mutex contention on System methods
            struct AtomData {
                pos: Vec3,
                mass: f32,
                atomic_number: u8,
                chain_id: String,
                residue_name: String,
                residue_id: u32,
                atom_name: String,
            }
            
            let parsed_atoms: Vec<AtomData> = lines[atom_lines_start..atom_lines_end]
                .par_iter()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') { return None; }
                    
                    // Simple Splitter (Optimized for standard CIF, quote aware)
                    // Copying logic from before but into closure
                    let mut parts = Vec::new();
                    let mut current_part = String::new();
                    let mut in_quote = false;
                    let mut quote_char = ' ';
                    
                    for c in trimmed.chars() {
                        if in_quote {
                            if c == quote_char { in_quote = false; } else { current_part.push(c); }
                        } else if c == '\'' || c == '"' {
                            in_quote = true; quote_char = c;
                        } else if c.is_whitespace() {
                            if !current_part.is_empty() {
                                parts.push(current_part.clone());
                                current_part.clear();
                            }
                        } else {
                            current_part.push(c);
                        }
                    }
                    if !current_part.is_empty() { parts.push(current_part); }
                    
                    // Indexing
                    let x_idx = col_indices.get("_atom_site.Cartn_x")?;
                    let y_idx = col_indices.get("_atom_site.Cartn_y")?;
                    let z_idx = col_indices.get("_atom_site.Cartn_z")?;
                    
                    if parts.len() <= *z_idx.max(x_idx).max(y_idx) { return None; }
                    
                    let x: f32 = parts[*x_idx].parse().ok()?;
                    let y: f32 = parts[*y_idx].parse().ok()?;
                    let z: f32 = parts[*z_idx].parse().ok()?;
                    
                    let element_str = col_indices.get("_atom_site.type_symbol").and_then(|&i| parts.get(i)).map(|s| s.as_str()).unwrap_or("C");
                    let atomic_number = match element_str {
                        "H" => 1, "C" => 6, "N" => 7, "O" => 8, "S" => 16, "P" => 15, _ => 6
                    };
                    let mass = match atomic_number { 1=>1.008, 6=>12.01, 7=>14.01, 8=>16.0, 15=>30.97, 16=>32.06, _=>12.0 };
                     
                    let chain_id = col_indices.get("_atom_site.label_asym_id")
                        .or_else(|| col_indices.get("_atom_site.auth_asym_id"))
                        .and_then(|&i| parts.get(i)).cloned().unwrap_or_else(|| "A".to_string());
                        
                    let residue_name = col_indices.get("_atom_site.label_comp_id")
                         .or_else(|| col_indices.get("_atom_site.auth_comp_id"))
                         .and_then(|&i| parts.get(i)).cloned().unwrap_or_else(|| "UNK".to_string());

                    let residue_id = col_indices.get("_atom_site.label_seq_id")
                        .or_else(|| col_indices.get("_atom_site.auth_seq_id"))
                        .and_then(|&i| parts.get(i))
                        .and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);

                     let atom_name = col_indices.get("_atom_site.label_atom_id")
                        .or_else(|| col_indices.get("_atom_site.auth_atom_id"))
                        .and_then(|&i| parts.get(i)).cloned().unwrap_or_else(|| "UNK".to_string());

                    Some(AtomData { pos: Vec3::new(x,y,z), mass, atomic_number, chain_id, residue_name, residue_id, atom_name })
                })
                .collect();
                
            // Serial addition to system (fast)
            for atom in parsed_atoms {
                min = min.min(atom.pos);
                max = max.max(atom.pos);
                // SS is Loop by default, calculated later or effectively ignored here
                system.add_atom(atom.pos, atom.mass, atom.atomic_number, atom.chain_id, atom.residue_name, atom.residue_id, SecondaryStructure::Loop, atom.atom_name);
            }
        }

        
        if system.positions.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No atomic coordinates found in CIF file. This may be a reference file without coordinates."
            ));
        }
        
        // Auto-center box
        let size = max - min;
        system.box_size = size + 10.0;
        let center = (min + max) * 0.5;
        let box_center = system.box_size * 0.5;
        let offset = box_center - center;
        for pos in &mut system.positions {
            *pos += offset;
        }
        
        // Infer bonds
        // Optimization: For large systems (>100k atoms), skip bond inference to speed up loading and avoid OOM.
        if system.positions.len() <= 100_000 {
            let cutoff_sq = 2.2 * 2.2;
            let cell_size = 2.5;
            let mut grid = SpatialGrid::new(system.box_size, cell_size);
            grid.insert(&system);
        
            let (dim_x, dim_y, dim_z) = grid.dimensions;

        for i in 0..system.positions.len() {
            let pos = system.positions[i];
            let cx = (pos.x / cell_size).floor() as i32;
            let cy = (pos.y / cell_size).floor() as i32;
            let cz = (pos.z / cell_size).floor() as i32;
            
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        let nz = cz + dz;
                        if nx >= 0 && nx < dim_x as i32 && ny >= 0 && ny < dim_y as i32 && nz >= 0 && nz < dim_z as i32 {
                            let cell_idx = (nx as usize) + (ny as usize) * dim_x + (nz as usize) * dim_x * dim_y;
                            if cell_idx < grid.cells.len() {
                                for &j in &grid.cells[cell_idx] {
                                    if (i as u32) < j {
                                        let diff = pos - system.positions[j as usize];
                                        let dist_sq = diff.length_squared();
                                        if dist_sq < cutoff_sq {
                                            system.bonds.push((i as u32, j, BondOrder::Single));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            }
        } else {
             println!("Large CIF system detected ({} atoms). Skipping bond inference.", system.positions.len());
        }
        
        // Auto-detect molecule information
        system.auto_detect_molecule_info();
        
        println!("Inferred {} bonds from {} atoms (CIF)", system.bonds.len(), system.positions.len());
        println!("Molecule Type: {:?}, Classification: {:?}", system.molecule_info.mol_type, system.molecule_info.classification);
        if let Some(ref formula) = system.molecule_info.formula {
            println!("Formula: {}", formula);
        }
        println!("Found {} Helix ranges and {} Sheet ranges (CIF)", helices.len(), sheets.len());
        Ok(system)
    }
}

pub struct SpatialGrid {
    pub cell_size: f32,
    pub dimensions: (usize, usize, usize),
    pub cells: Vec<Vec<u32>>,
}

impl SpatialGrid {
    pub fn new(box_size: Vec3, cell_size: f32) -> Self {
        let x = (box_size.x / cell_size).ceil() as usize;
        let y = (box_size.y / cell_size).ceil() as usize;
        let z = (box_size.z / cell_size).ceil() as usize;
        Self {
            cell_size,
            dimensions: (x, y, z),
            cells: vec![Vec::new(); x * y * z],
        }
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    pub fn insert(&mut self, system: &System) {
        self.clear();
        for (i, pos) in system.positions.iter().enumerate() {
            let x = (pos.x / self.cell_size).floor() as usize;
            let y = (pos.y / self.cell_size).floor() as usize;
            let z = (pos.z / self.cell_size).floor() as usize;
            
            // Boundary checks
            if x < self.dimensions.0 && y < self.dimensions.1 && z < self.dimensions.2 {
                let idx = x + y * self.dimensions.0 + z * self.dimensions.0 * self.dimensions.1;
                self.cells[idx].push(i as u32);
            }
        }
    }
}

impl System {
    pub fn initialize_maxwell_boltzmann(&mut self, temperature: f32) {
        // Simple random velocity initialization
        // v = sqrt(k_B * T / m) * Gaussian(0, 1)
        // kB ~ 8.617e-5 eV/K.
        let k_b = 8.617e-5; 
        
        for i in 0..self.velocities.len() {
            let m = if i < self.masses.len() { self.masses[i] } else { 12.0 };
            if m <= 0.0 { continue; }
            
            let std_dev = (k_b * temperature / m).sqrt();
            
            // Simple Linear Congruential Generator for deterministic "randomness"
            let vx = (self.pseudo_rand(i as u32) - 0.5) * std_dev * 2.0;
            let vy = (self.pseudo_rand(i as u32 + 10000) - 0.5) * std_dev * 2.0;
            let vz = (self.pseudo_rand(i as u32 + 20000) - 0.5) * std_dev * 2.0;
            
            self.velocities[i] = Vec3::new(vx, vy, vz);
        }
    }

    fn pseudo_rand(&self, seed: u32) -> f32 {
        let mut x = seed;
        x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
        x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
        let x = (x >> 16) ^ x;
        (x as f32) / (u32::MAX as f32)
    }
}

impl System {
    /// Unified Loader: Auto-detects file format by extension
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let path_ref = path.as_ref();
        let extension = path_ref.extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(|s| s.to_lowercase())
            .unwrap_or_else(|| "".to_string());

        match extension.as_str() {
            "pdb" | "ent" => Self::from_pdb(path),
            "cif" | "mmcif" => Self::from_cif(path),
            "xyz" => Self::from_xyz(path),
            "sdf" | "mol" => Self::from_sdf(path),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput, 
                format!("Unsupported file extension: .{}", extension)
            ))
        }
    }

    pub fn from_sdf<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.len() < 4 { return Ok(Self::new(0)); }
        
        // Line 4 (index 3) is Counts Line
        let counts_line = lines[3];
        let n_atoms: usize;
        let n_bonds: usize;
        
        if let (Ok(na), Ok(nb)) = (counts_line[0..3].trim().parse(), counts_line[3..6].trim().parse()) {
            n_atoms = na;
            n_bonds = nb;
        } else {
             let parts: Vec<&str> = counts_line.split_whitespace().collect();
             n_atoms = parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(0);
             n_bonds = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        }
        
        let mut sys = Self::new(n_atoms);
        
        for i in 0..n_atoms {
            if 4+i >= lines.len() { break; }
            let line = lines[4+i];
            if line.len() < 34 { continue; }
            
            let x: f32 = line[0..10].trim().parse().unwrap_or(0.0);
            let y: f32 = line[10..20].trim().parse().unwrap_or(0.0);
            let z: f32 = line[20..30].trim().parse().unwrap_or(0.0);
            let symbol = line[31..34].trim();
            
            sys.positions.push(glam::Vec3::new(x, y, z));
            sys.atomic_numbers.push(symbol_to_atomic_number(symbol));
            sys.atom_names.push(symbol.to_string());
            sys.residue_ids.push(1);
            sys.chain_ids.push("A".to_string());
            sys.residue_names.push("MOL".to_string());
            sys.masses.push(get_mass_for_element(symbol_to_atomic_number(symbol)));
            sys.velocities.push(glam::Vec3::ZERO);
            sys.forces.push(glam::Vec3::ZERO);
            sys.ids.push(i as u32);
            sys.secondary_structure.push(SecondaryStructure::Unknown);
        }
        
        for i in 0..n_bonds {
            if 4 + n_atoms + i >= lines.len() { break; }
            let line = lines[4 + n_atoms + i];
            
            // SDF Bond Block:
            // 111 222 333 444 555 666 777
            // 111 = 1st atom index
            // 222 = 2nd atom index
            // 333 = Bond Type (1=Single, 2=Double, 3=Triple, 4=Aromatic)
            
            let a1: u32 = line[0..3].trim().parse().unwrap_or(0);
            let a2: u32 = line[3..6].trim().parse().unwrap_or(0);
            let type_code: u8 = line[6..9].trim().parse().unwrap_or(1); // Default to single
            
            let order = match type_code {
                1 => BondOrder::Single,
                2 => BondOrder::Double,
                3 => BondOrder::Triple,
                4 => BondOrder::Aromatic,
                _ => BondOrder::Single, // Default/Unknown
            };
            
            if a1 > 0 && a2 > 0 {
                sys.bonds.push(((a1 - 1) as u32, (a2 - 1) as u32, order));
            }
        }
        
        sys.auto_detect_molecule_info();
        if let Some(name) = lines.first() {
            if !name.trim().is_empty() {
                sys.molecule_info.name = Some(name.trim().to_string());
            }
        }
        
        // Recenter
        if !sys.positions.is_empty() {
            let mut min = sys.positions[0];
            let mut max = sys.positions[0];
            for &p in &sys.positions {
                min = min.min(p);
                max = max.max(p);
            }
             // Update Box Size & Center logic
            let size = max - min;
            sys.box_size = size + 20.0;
            let center = (min + max) * 0.5;
            let box_center = sys.box_size * 0.5;
            let shift = box_center - center;
            for p in &mut sys.positions { *p += shift; }
        }
        
        Ok(sys)
    }
}

fn symbol_to_atomic_number(s: &str) -> u8 {
    match s.trim().to_uppercase().as_str() {
        "H" => 1, "HE" => 2, "LI" => 3, "BE" => 4, "B" => 5, "C" => 6, "N" => 7, "O" => 8, "F" => 9, "NE" => 10,
        "NA" => 11, "MG" => 12, "AL" => 13, "SI" => 14, "P" => 15, "S" => 16, "CL" => 17, "AR" => 18,
        "K" => 19, "CA" => 20, "FE" => 26, "CU" => 29, "ZN" => 30, "BR" => 35, "AG" => 47, "I" => 53, "AU" => 79,
        "HG" => 80, "PB" => 82, "U" => 92,
        _ => 6
    }
}

pub fn get_mass_for_element(atomic_number: u8) -> f32 {
    match atomic_number {
        1 => 1.008,
        2 => 4.0026,
        6 => 12.011,
        7 => 14.007,
        8 => 15.999,
        9 => 18.998,
        15 => 30.974,
        16 => 32.06,
        17 => 35.45,
        26 => 55.845,
        _ => 12.0,
    }
}

// --- Graph Algorithms for Selection ---
pub type AdjacencyList = Vec<Vec<usize>>;

impl System {
    /// Builds an adjacency list from the current bonds.
    /// O(B) where B is number of bonds.
    pub fn build_adjacency_list(&self) -> AdjacencyList {
        let mut adj = vec![vec![]; self.positions.len()];
        for &(a, b, _) in &self.bonds {
             let a = a as usize;
             let b = b as usize;
             if a < adj.len() && b < adj.len() {
                 adj[a].push(b);
                 adj[b].push(a);
             }
        }
        adj
    }

    /// Finds all atoms connected to the start_atom using BFS.
    /// Returns a list of atom indices forming the connected component (molecule/chain).
    pub fn find_connected_component(&self, start_atom: usize, adj: &AdjacencyList) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut component = Vec::new();

        if start_atom >= self.positions.len() {
            return component;
        }

        queue.push_back(start_atom);
        visited.insert(start_atom);

        while let Some(current) = queue.pop_front() {
            component.push(current);
            if current < adj.len() {
                for &neighbor in &adj[current] {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        component
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_ligand_identification() {
        let mut system = System::new(10);
        // Create 2 separate molecules
        // Mol A: 0-1-2
        system.positions.extend(vec![Vec3::ZERO; 3]);
        system.bonds.push((0, 1, BondOrder::Single));
        system.bonds.push((1, 2, BondOrder::Single));

        // Mol B: 3-4
        system.positions.extend(vec![Vec3::ZERO; 2]);
        system.bonds.push((3, 4, BondOrder::Single));

        // Isolated Atom: 5
        system.positions.push(Vec3::ZERO);

        let adj = system.build_adjacency_list();
        
        let mol_a = system.find_connected_component(0, &adj);
        assert_eq!(mol_a.len(), 3);
        assert!(mol_a.contains(&0));
        assert!(mol_a.contains(&1));
        assert!(mol_a.contains(&2));
        assert!(!mol_a.contains(&3));

        let mol_b = system.find_connected_component(3, &adj);
        assert_eq!(mol_b.len(), 2);
        assert!(mol_b.contains(&3));
        assert!(mol_b.contains(&4));
        
        let mol_c = system.find_connected_component(5, &adj);
        assert_eq!(mol_c.len(), 1);
        assert!(mol_c.contains(&5));
    }
}
