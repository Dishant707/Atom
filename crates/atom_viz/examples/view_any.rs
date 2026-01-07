use atom_core::System;
use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let input_str = if args.len() > 1 {
        args[1].clone()
    } else {
        // Prompt user
        print!("Enter file path (or PDB ID '1UBQ', or Chemical Name 'Aspirin'):\n> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        input.trim().to_string()
    };

    if input_str.is_empty() {
        println!("No input. Starting empty. Drag & Drop a file!");
        let system = System::new(100);
        pollster::block_on(atom_viz::run(system, None, std::collections::HashMap::new()));
        return;
    }

    let path = Path::new(&input_str);
    let mut final_path = input_str.clone();

    // 1. Check if local file exists
    if !path.exists() {
         println!("'{}' not found locally. Searching online providers...", input_str);
         
         // 2. Check if PDB ID (4 alphanumeric)
         if input_str.len() == 4 && input_str.chars().all(|c| c.is_ascii_alphanumeric()) {
             let pdb_id = input_str.to_uppercase();
             let url = format!("https://files.rcsb.org/download/{}.pdb", pdb_id);
             let output_file = format!("{}.pdb", pdb_id);
             
             println!("Fetching PDB {} from RCSB...", pdb_id);
             let status = Command::new("curl")
                .args(&["-L", "-f", "-o", &output_file, &url])
                .status();
                
             if let Ok(s) = status {
                 if s.success() {
                     println!("Downloaded {}.pdb", pdb_id);
                     final_path = output_file;
                 } else {
                     eprintln!("Failed to download PDB {} (likely too large or 404). Trying CIF format...", pdb_id);
                     let cif_url = format!("https://files.rcsb.org/download/{}.cif.gz", pdb_id);
                     let cif_gz_output = format!("{}.cif.gz", pdb_id);
                     let cif_output = format!("{}.cif", pdb_id);
                     
                     println!("Fetching compressed CIF {} from RCSB...", pdb_id);
                     let cif_status = Command::new("curl")
                        .args(&["--retry", "5", "--retry-delay", "1", "--http1.1", "-L", "-f", "-o", &cif_gz_output, &cif_url])
                        .status();
                        
                     let download_success = if let Ok(s) = cif_status { s.success() } else { false };
                     
                     if !download_success {
                         eprintln!("Curl failed. Trying Python fallback...");
                         let py_status = Command::new("python3")
                             .args(&["crates/atom_viz/examples/download_helper.py", &pdb_id])
                             .status();
                             
                         if let Ok(ps) = py_status {
                             if ps.success() {
                                 // Python script handles decompression too
                                 final_path = cif_output;
                             } else {
                                 eprintln!("Python download also failed.");
                             }
                         }
                     } else {
                         // Curl success, check decompression
                         println!("Downloaded {}.cif.gz. Decompressing...", pdb_id);
                         let unzip_status = Command::new("gzip")
                             .args(&["-d", "-f", &cif_gz_output])
                             .status();
                             
                         if let Ok(us) = unzip_status {
                             if us.success() {
                                 println!("Decompressed to {}.cif", pdb_id);
                                 final_path = cif_output;
                             } else {
                                 eprintln!("Failed to decompress {}.cif.gz", pdb_id);
                             }
                         }
                     }
                 }
             }
         } else {
             // 3. Assume Chemical Name -> PubChem
             let name = input_str.clone();
             // Encode spaces? Simple implementation assumes single word or user handles it, 
             // but let's just try direct string.
             let url = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/SDF", name);
             let sdf_file = format!("{}.sdf", name);
             let pdb_file = format!("{}.pdb", name);
             
             println!("Fetching '{}' from PubChem...", name);
             let status = Command::new("curl")
                .args(&["-L", "-f", "-o", &sdf_file, &url])
                .status();
                
             if let Ok(s) = status {
                 if s.success() {
                      // Native Ingest
                      println!("Downloaded {}.sdf. Loading native...", name);
                      final_path = sdf_file;
                 } else {
                      eprintln!("Could not find '{}' on PubChem.", name);
                 }
             }
         }
    }

    // Try loading again with final path
    let final_path_obj = Path::new(&final_path);
    if !final_path_obj.exists() {
         println!("Could not load '{}'. Starting empty.", final_path);
         let system = System::new(100);
         pollster::block_on(atom_viz::run(system, None, std::collections::HashMap::new()));
         return;
    }

    println!("Loading {}...", final_path);
    match System::load_from_file(final_path_obj) {
        Ok(system) => {
            println!("Successfully loaded system with {} atoms.", system.positions.len());
            pollster::block_on(atom_viz::run(system, None, std::collections::HashMap::new()));
        }
        Err(e) => {
            eprintln!("Error loading file: {}", e);
        }
    }
}
