use atom_core::potential::Potential;
use anyhow::Result;

fn main() -> Result<()> {
    // Hardcoded path to the uploaded model
    let model_path = "/Users/dishant/Desktop/Atom/ani2x_complete.onnx";
    
    println!("Loading ANI-2x model from: {}", model_path);
    let mut potential = Potential::new(model_path)?;
    
    // Methane (CH4)
    // Species: C=6, H=1
    // ANI mapping might be generic atomic numbers or mapped indices (0..6).
    // The user's colab_export.py showed:
    // self.elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
    // And input was: species = torch.tensor([[1, 0, 0, 0, 0]]) # CH4
    // So 0=H, 1=C.
    // Wait, let's verify the mapping in colab_export.py.
    
    /*
    class ReconstructedANI(nn.Module):
        ...
        self.elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
    */
    // If the model expects indices: H=0, C=1, N=2, O=3...
    // The previous export code used: species = torch.tensor([[1, 0, 0, 0, 0]])
    // This implies C(1) and H(0).
    
    let species = vec![1, 0, 0, 0, 0]; // C, H, H, H, H
    
    // Approximate Methane coordinates (Angstroms)
    let coordinates = vec![
        0.0000, 0.0000, 0.0000,  // C
        0.6300, 0.6300, 0.6300,  // H
       -0.6300,-0.6300, 0.6300,  // H
       -0.6300, 0.6300,-0.6300,  // H
        0.6300,-0.6300,-0.6300   // H
    ];
    
    println!("Computing Energy and Forces for Methane...");
    let (energy, forces) = potential.compute(&species, &coordinates)?;
    
    println!("Energy: {} Hartree", energy);
    println!("Forces:");
    for (i, f) in forces.chunks(3).enumerate() {
        println!("  Atom {}: [{:.4}, {:.4}, {:.4}]", i, f[0], f[1], f[2]);
    }
    
    Ok(())
}
