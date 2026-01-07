
import sys

def parse_fasta(file_path):
    seq = ""
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return seq

def generate_linear_xyz(seq, output_path):
    # Mapping of Amino Acid (1-letter) to approx atom count or just Backbone?
    # Let's just generate Backbone (C-alpha) for simplicity? 
    # Or All-Atom (approx)?
    # To keep it simple and standard for ANI (which expects H, C, N, O), 
    # we will just generate a "bead" model or simplified backbone.
    # But ANI needs atomic species (1, 6, 7, 8).
    # Let's generate a simple backbone: N - C_alpha - C - O ...
    
    # Standard bond length approx 1.5A
    
    with open(output_path, 'w') as f:
        # We will write an XYZ file.
        # First pass to count atoms.
        # 4 atoms per residue (N, CA, C, O) + side chain?
        # Let's do pure backbone (Glycine-like) to start: N-C-C=O
        
        atoms = []
        x = 0.0
        
        # Simple extended chain along X axis
        for res in seq:
            # N
            atoms.append(f"N {x} 0.0 0.0")
            x += 1.45
            # CA
            atoms.append(f"C {x} 0.0 0.0")
            x += 1.52
            # C
            atoms.append(f"C {x} 0.0 0.0")
            # O (attached to C)
            atoms.append(f"O {x} 1.23 0.0")
            x += 1.33
            
        f.write(f"{len(atoms)}\n")
        f.write(f"Generated from FASTA: {seq[:20]}...\n")
        for atom in atoms:
            f.write(f"{atom}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fasta_to_xyz.py <fasta_file>")
        sys.exit(1)
        
    fasta_file = sys.argv[1]
    seq = parse_fasta(fasta_file)
    print(f"Read Sequence Length: {len(seq)}")
    
    out_file = fasta_file + ".xyz"
    generate_linear_xyz(seq, out_file)
    print(f"Generated 3D Structure: {out_file}")
