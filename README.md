
# Atom ⚛️
**AI-Powered Molecular Dynamics Engine**

Atom is a high-performance Molecular Dynamics (MD) simulation engine written in Rust. It leverages Deep Neural Networks (ANI-2x) to calculate inter-atomic forces with quantum-mechanical accuracy, bypassing the need for traditional classical force fields.

## 🚀 Key Features

- **Neural Network Physics**: Directly uses ANI-2x ONNX models to calculate Potential Energy and Forces.
- **Hybrid Engine**:
  - **Turbo Mode**: Uses Analytical Gradients from the AI model (1000x faster).
  - **Compatibility Mode**: Uses Finite Difference with Massive Parallel Batching for models without baked gradients.
- **Hardware Acceleration**:
  - **Apple Silicon**: Native CoreML and Metal support via `ort`.
  - **CPU**: Optimized AVX/SIMD execution on Intel/AMD.
- **Real-Time Visualization**:
  - Interactive 3D rendering (WGPU).
  - Thermal Dynamics (Maxwell-Boltzmann Initialization).
  - Interactive Bond Breaking/Formation.
- **File Support**:
  - `.pdb` (Protein Data Bank)
  - `.xyz` (Standard Cartesian)
  - `.fasta` (Amino Acid Sequence -> 3D Chain)

## 🛠️ Architecture

The project is a Rust Workspace split into two crates:

### 1. `crates/atom_core`
The brain of the operation.
- **System**: Manages Atoms, Positions, Velocities, Caches.
- **Potential**: Handles ONNX Runtime (ORT) sessions, batches inputs, and computes forces.
- **Integrator**: Velocity Verlet algorithm for time-stepping.

### 2. `crates/atom_viz`
The eyes.
- **Renderer**: WGPU-based 3D renderer.
- **Controls**: Orbit, Pan, Zoom, Pivot-to-Atom (`G`).
- **UI**: Egui overlay for stats.

## 📦 Getting Started

### Prerequisites
- [Rust Toolchain](https://rustup.rs/) (Stable)
- [ONNX Runtime](https://onnxruntime.ai/) (Binaries handled by `ort` crate, but MacOS needs lib copy)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/atom.git
   cd atom
   ```

2. **Setup Dependencies (MacOS)**
   ONNX Runtime dynamic libraries might need to be visible.
   ```bash
   bash setup_ort.sh
   export DYLD_LIBRARY_PATH=$(pwd)/target/release:$DYLD_LIBRARY_PATH
   ```

3. **Run the Simulation**
   To see a Protein (Crambin) folding in real-time:
   ```bash
   cargo run --release --example view_pdb -- molecules/1crn.pdb
   ```

   To simulate a custom protein from sequence:
   ```bash
   # Convert FASTA to Linear Chain
   python3 scripts/fasta_to_xyz.py my_protein.fasta
   # Run
   cargo run --release --example view_pdb -- my_protein.fasta.xyz
   ```

## 🧠 Model Configuration

Atom searches for `ani2x_complete.onnx` or `physics_model.onnx` in the root directory.

- **Default**: The engine loads the model.
- **Optimization**: If the model provides `forces` output, Atom uses it. If not, it falls back to Finite Difference (slower).

## 🎮 Controls

- **Left Mouse**: Rotate View
- **Right Mouse**: Pan
- **Scroll**: Zoom (Speed scales with distance)
- **G**: Grip (Pivot camera around the atom under cursor)
- **Space**: Pause/Resume Simulation
- **R**: Reset Camera

