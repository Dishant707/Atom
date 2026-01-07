import re
import os
import sys

def main():
    print("Reading table_snippet.txt...")
    try:
        with open("table_snippet.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: table_snippet.txt not found")
        sys.exit(1)

    # Clean header/footer to extract just the array initialization
    # Remove the variable declaration
    if "int triTable[256][16] =" in text:
        text = text.split("int triTable[256][16] =")[1]
    
    # We only want content up to the closing brace and semicolon
    if "};" in text:
        text = text.split("};")[0]
    
    # Replace braces and commas with spaces to ease parsing
    cleaned = text.replace("{", " ").replace("}", " ").replace(",", " ")
    
    # Parse integers
    numbers = []
    for token in cleaned.split():
        token = token.strip()
        if not token: continue
        try:
            numbers.append(int(token))
        except ValueError:
            pass
            
    expected_len = 256 * 16
    print(f"Parsed {len(numbers)} integers.")
    
    if len(numbers) != expected_len:
        print(f"CRITICAL ERROR: Data mismatch. Expected {expected_len}, got {len(numbers)}.")
        print("Aborting update to prevent corruption.")
        sys.exit(1)

    # Format as Rust array string
    print("Formatting Rust array...")
    lines = []
    for i in range(0, len(numbers), 16):
        chunk = numbers[i:i+16]
        line = ", ".join(str(n) for n in chunk)
        lines.append(f"    {line}")
    
    rust_table_content = ",\n".join(lines)
    full_rust_code = f"const TRI_TABLE: [i8; 256 * 16] = [\n{rust_table_content}\n];"

    # Update surface.rs
    target_file = "crates/atom_core/src/surface.rs"
    print(f"Updating {target_file}...")
    
    with open(target_file, "r") as f:
        file_content = f.read()
        
    # Regex to find the existing constant definition
    # Matches: const TRI_TABLE: [i8; 256 * 16] = [ ... ];
    # Use dotall to match newlines
    pattern = re.compile(r"const TRI_TABLE: \[i8; 256 \* 16\] = \[\s*.*?\];", re.DOTALL)
    
    if not pattern.search(file_content):
        print("Error: Could not find existing TRI_TABLE definition in surface.rs")
        sys.exit(1)
        
    new_content = pattern.sub(full_rust_code, file_content)
    
    # Fix Types
    new_content = new_content.replace("(val2 - val1).abs() < 1e-5", "(val2 - val1).abs() < 1e-5_f32")
    
    # Fix Imports
    new_content = new_content.replace("use glam::{Vec3, Vec3Swizzles};", "use glam::Vec3;")
    
    with open(target_file, "w") as f:
        f.write(new_content)
        
    print("Success! File updated.")

if __name__ == "__main__":
    main()
