use crate::{State, RenderMode};

pub fn get_element_name(atomic_number: u8) -> &'static str {
    match atomic_number {
        1 => "Hydrogen",
        2 => "Helium",
        6 => "Carbon",
        7 => "Nitrogen",
        8 => "Oxygen",
        9 => "Fluorine",
        11 => "Sodium",
        12 => "Magnesium",
        15 => "Phosphorus",
        16 => "Sulfur",
        17 => "Chlorine",
        19 => "Potassium",
        20 => "Calcium",
        26 => "Iron",
        _ => "Unknown",
    }
}

pub fn get_element_color(atomic_number: u8) -> egui::Color32 {
    match atomic_number {
        1 => egui::Color32::WHITE,
        6 => egui::Color32::from_gray(80), // Dark Grey
        7 => egui::Color32::BLUE,
        8 => egui::Color32::RED,
        15 => egui::Color32::from_rgb(255, 165, 0), // Orange
        16 => egui::Color32::YELLOW,
        _ => egui::Color32::from_rgb(255, 0, 255), // Magenta
    }
}

// Helper to project 3D point to 2D screen
fn project_point(pos: glam::Vec3, view_proj: glam::Mat4, screen_size: egui::Rect) -> Option<egui::Pos2> {
    let ndc = view_proj * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
    // Clipped?
    if ndc.w <= 0.0 { return None; }
    let ndc_point = ndc.truncate() / ndc.w;
    // Check if outside screen bounds (optional, but good for performance)
    // if ndc_point.x < -1.0 || ndc_point.x > 1.0 || ndc_point.y < -1.0 || ndc_point.y > 1.0 { return None; }

    let x = (ndc_point.x + 1.0) * 0.5 * screen_size.width();
    let y = (1.0 - ndc_point.y) * 0.5 * screen_size.height(); // Flip Y
    Some(egui::Pos2::new(x + screen_size.min.x, y + screen_size.min.y))
}

pub fn draw_ui(state: &mut State, ctx: &egui::Context) {
    // Render 3D Distance Labels
    let screen_rect = ctx.screen_rect();
    let view_proj = state.camera.build_view_projection_matrix();
    
    // Create a painter for the background layer
    let painter = ctx.layer_painter(egui::LayerId::background());

    for (a, b, dist) in &state.measurements {
        if *a < state.system.positions.len() && *b < state.system.positions.len() {
             let p1 = state.system.positions[*a];
             let p2 = state.system.positions[*b];
             let mid = (p1 + p2) * 0.5;
             
             if let Some(screen_pos) = project_point(mid, view_proj, screen_rect) {
                 painter.text(
                     screen_pos,
                     egui::Align2::CENTER_CENTER,
                     format!("{:.2} Å", dist),
                     egui::FontId::default(),
                     egui::Color32::YELLOW,
                 );
                 // Optional: Draw line? WGPU does it better for depth, but UI line is quick
                 /*
                 if let (Some(s1), Some(s2)) = (project_point(p1, view_proj, screen_rect), project_point(p2, view_proj, screen_rect)) {
                     painter.line_segment([s1, s2], egui::Stroke::new(1.0, egui::Color32::from_white_alpha(100)));
                 }
                 */
             }
        }
    }


    egui::TopBottomPanel::bottom("sequence_panel").min_height(40.0).show(ctx, |ui| {
         ui.horizontal(|ui| {
             ui.label(egui::RichText::new("Sequence:").strong());
             if state.residue_cache.is_empty() {
                 ui.label("No residue data available.");
             } else {
                 egui::ScrollArea::horizontal().show(ui, |ui| {
                     for res in &state.residue_cache {
                         let label = format!("{} {}", res.name, res.id);
                         let mut is_selected = false;
                         if let Some(sel) = state.selected_atom {
                              if res.atom_range.contains(&sel) { is_selected = true; }
                         }
                         
                         let btn = ui.add(egui::Button::new(label)
                             .selected(is_selected)
                             .fill(if is_selected { egui::Color32::DARK_GREEN } else { egui::Color32::from_gray(40) }));
                             
                         if btn.clicked() {
                             // Select Residue
                             state.measurement_mode = false; // Turn off measurement
                             state.selected_atom = Some(res.atom_range.start);
                             state.selected_component = res.atom_range.clone().collect(); // Range to HashSet
                             state.needs_buffer_update = true;
                             
                             // Focus Camera
                             let mut center = glam::Vec3::ZERO;
                             let mut count = 0.0;
                             for i in res.atom_range.clone() {
                                 if i < state.system.positions.len() {
                                     center += state.system.positions[i];
                                     count += 1.0;
                                 }
                             }
                             if count > 0.0 { 
                                 center /= count; 
                                 state.camera.target = center;
                                 let dir = (state.camera.eye - state.camera.target).normalize();
                                 state.camera.eye = center + dir * 20.0;
                             }
                         }
                         
                         if btn.hovered() {
                             ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                         }
                     }
                 });
             }
         });
    });

    egui::TopBottomPanel::bottom("mol_info_panel").show(ctx, |ui| {
        if state.render_mode == RenderMode::Ribbon && !state.system.has_structural_metadata() {
             ui.colored_label(egui::Color32::YELLOW, "⚠ Warning: No HELIX/SHEET records found. Ribbons rendered as trace.");
             ui.separator();
        }
        ui.horizontal(|ui| {
             ui.label(egui::RichText::new("Molecule:").strong());
             if let Some(name) = &state.system.molecule_info.name {
                  ui.label(egui::RichText::new(name).color(egui::Color32::LIGHT_BLUE));
             } else {
                  ui.label("Unknown");
             }
             ui.separator();

             ui.label(egui::RichText::new("Formula:").strong());
             if let Some(formula) = &state.system.molecule_info.formula {
                  ui.label(formula);
             } else {
                  ui.label("?");
             }
             ui.separator();

             ui.label(egui::RichText::new("Class:").strong());
             ui.label(format!("{:?}", state.system.molecule_info.classification));
        });
    });

    egui::Window::new("Atom Inspector")
        .default_open(true)
        .show(ctx, |ui| {

            ui.collapsing("Selection & Focus", |ui| {
                if let Some(idx) = state.selected_atom {
                    let z = state.system.atomic_numbers[idx];
                    let name = get_element_name(z);
                    ui.label(egui::RichText::new(format!("Selected Atom: #{} {} ({})", idx, name, z)).color(egui::Color32::GREEN).strong());
                    
                    // Show Residue Info if available
                    if idx < state.system.residue_names.len() {
                         let res = &state.system.residue_names[idx];
                         let res_id = if idx < state.system.residue_ids.len() { state.system.residue_ids[idx].to_string() } else { "?".to_string() };
                         let chain = if idx < state.system.chain_ids.len() { &state.system.chain_ids[idx] } else { "?" };
                         ui.label(format!("Residue: {} {} (Chain {})", res, res_id, chain));
                    }
                    
                    if ui.button("🎯 Focus Camera").clicked() {
                        // ... existing focus logic ...
                         let target = state.system.positions[idx];
                         let mut center = target;
                         let _count = 1.0;
                         // ...
                         if let Some(_color) = state.custom_colors.get(&idx) {
                             let mut sum = glam::Vec3::ZERO;
                             let mut c = 0.0;
                             for (id, _) in &state.custom_colors {
                                 if *id < state.system.positions.len() {
                                     sum += state.system.positions[*id];
                                     c += 1.0;
                                 }
                             }
                             if c > 0.0 { center = sum / c; }
                         }
                         
                         state.camera.target = center;
                         let dir = (state.camera.eye - state.camera.target).normalize(); 
                         state.camera.eye = center + dir * 20.0; 
                    }
                    
                    if ui.checkbox(&mut state.is_isolated, "👁 Isolate Selection").changed() {
                        state.needs_buffer_update = true;
                    }

                } else {
                    ui.label("No Selection (Click an atom)");
                }
            });

            ui.collapsing("Visualization", |ui| {
                 ui.horizontal(|ui| {
                     ui.label("Atom Scale");
                     if ui.add(egui::Slider::new(&mut state.atom_scale, 0.1..=2.0)).changed() {
                         state.needs_buffer_update = true;
                     }
                 });
                 ui.horizontal(|ui| {
                     ui.label("Bond Radius");
                     if ui.add(egui::Slider::new(&mut state.bond_radius, 0.05..=0.5)).changed() {
                         state.needs_buffer_update = true;
                     }
                 });
                 ui.checkbox(&mut state.show_forces, "Show Forces (Green)");
                 if ui.button("📸 Screenshot").clicked() {
                     state.save_screenshot();
                 }
            });
            


            ui.collapsing("Measurements", |ui| {
                ui.checkbox(&mut state.measurement_mode, "📏 Measurement Mode");
                if state.measurement_mode {
                     ui.label(egui::RichText::new("Click 2 atoms to measure distance.").italics());
                     if !state.measurement_click_buffer.is_empty() {
                         ui.label(egui::RichText::new("Select 2nd point...").color(egui::Color32::YELLOW));
                     }
                }
                
                if !state.measurements.is_empty() {
                    ui.separator();
                    ui.label(format!("{} active measurements", state.measurements.len()));
                    if ui.button("Clear All").clicked() {
                        state.measurements.clear();
                        state.measurement_click_buffer.clear();
                    }
                }
            });

            ui.collapsing("Composition", |ui| {
                 // Element Counts
                 let mut counts = std::collections::HashMap::new();
                 for &z in &state.system.atomic_numbers {
                     *counts.entry(z).or_insert(0) += 1;
                 }
                 if counts.is_empty() {
                     ui.label("0 atoms");
                 } else {
                     let mut sorted: Vec<_> = counts.into_iter().collect();
                     sorted.sort_by_key(|a| a.0);
                     for (z, count) in sorted {
                         let name = get_element_name(z);
                         let color = get_element_color(z);
                         ui.colored_label(color, format!("{} ({}): {}", name, z, count));
                     }
                 }
            });

            // Map <(u8, u8), count>
            let mut bond_stats: std::collections::HashMap<(u8, u8), u32> = std::collections::HashMap::new();
            
            for &(i, j, _order) in &state.system.bonds {
                if (i as usize) < state.system.atomic_numbers.len() && (j as usize) < state.system.atomic_numbers.len() {
                    let a1 = state.system.atomic_numbers[i as usize];
                    let a2 = state.system.atomic_numbers[j as usize];
                    // Canonicalize (smaller first)
                    let key = if a1 < a2 { (a1, a2) } else { (a2, a1) };
                    *bond_stats.entry(key).or_insert(0) += 1;
                }
            }
            
            // Sort by count desc? or by type. Let's sort by count.
            let mut sorted_bonds: Vec<_> = bond_stats.into_iter().collect();
            sorted_bonds.sort_by(|a, b| b.1.cmp(&a.1));
            
            ui.collapsing(format!("Total Bonds: {}", state.system.bonds.len()), |ui| {
                  egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                    egui::Grid::new("bond_grid").striped(true).show(ui, |ui| {
                        for ((a1, a2), count) in sorted_bonds {
                            let n1 = get_element_name(a1);
                            let n2 = get_element_name(a2);
                            let c1 = get_element_color(a1);
                            let c2 = get_element_color(a2);
                            
                            ui.horizontal(|ui| {
                                ui.colored_label(c1, format!("{}", get_element_name(a1).chars().next().unwrap_or('?')));
                                ui.label("-");
                                ui.colored_label(c2, format!("{}", get_element_name(a2).chars().next().unwrap_or('?')));
                            });
                            ui.label(format!("({}-{})", n1, n2));
                            ui.label(format!("{}", count));
                            ui.end_row();
                        }
                    });
                  });
            });

            ui.separator();
            ui.label(format!("Total Atoms: {}", state.system.positions.len()));
            ui.label(format!("Total Bonds: {}", state.system.bonds.len()));
            ui.label(format!("Status: {}", if state.is_paused { "Paused" } else { "Running" }));
             if ui.button(if state.is_paused { "▶ Play (Space)" } else { "⏸ Pause (Space)" }).clicked() {
                 state.is_paused = !state.is_paused;
             }
            
            ui.separator();
            ui.heading("Quick Load");
            ui.horizontal(|ui| {
                 ui.text_edit_singleline(&mut state.load_input);
                if state.is_loading {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Loading...");
                    });
                } else {
                    if ui.button("Load").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                         let input = state.load_input.trim().to_string();
                         if !input.is_empty() {
                             state.is_loading = true;
                             let (tx, rx) = std::sync::mpsc::channel();
                             state.load_rx = Some(rx);
                             
                             std::thread::spawn(move || {
                                 let mut final_path = input.clone();
                                 let mut success = true;

                                 // 1. Check if Local File Exists
                                 if std::path::Path::new(&input).exists() {
                                     final_path = input.clone();
                                 
                                 // 2. Check PDB ID (4 chars)
                                 } else if input.len() == 4 && input.chars().all(|c| c.is_ascii_alphanumeric()) {
                                     let pdb_id = input.to_uppercase();
                                     let url = format!("https://files.rcsb.org/download/{}.pdb", pdb_id);
                                     let output_file = format!("{}.pdb", pdb_id);
                                     println!("Fetching PDB {} from {}...", pdb_id, url);
                                     
                                     let mut output = std::process::Command::new("curl")
                                        .args(&["-L", "-f", "-o", &output_file, &url])
                                        .output();
                                     
                                     // Check if PDB download failed, if so try CIF (common for large structures like 4V9D)
                                     // Check if PDB download failed, if so try CIF (common for large structures like 4V9D)
                                     if output.as_ref().map(|o| !o.status.success()).unwrap_or(true) {
                                          println!("PDB fetch failed, trying Robust CIF.gz strategy...");
                                          let cif_gz_url = format!("https://files.rcsb.org/download/{}.cif.gz", pdb_id);
                                          let cif_gz_file = format!("{}.cif.gz", pdb_id);
                                          let cif_file = format!("{}.cif", pdb_id);
                                          
                                          // 1. Try Curl with Retries & HTTP 1.1 (Download .gz)
                                          println!("Attempting curl fetch of {}...", cif_gz_file);
                                          output = std::process::Command::new("curl")
                                            .args(&["-L", "-f", "--http1.1", "--retry", "5", "--retry-delay", "1", "-o", &cif_gz_file, &cif_gz_url])
                                            .output();
                                            
                                          // 2. Python Fallback if Curl Fails
                                          if output.as_ref().map(|o| !o.status.success()).unwrap_or(true) {
                                              println!("Curl failed, trying Python fallback...");
                                              let script_path = "crates/atom_viz/examples/download_helper.py";
                                              println!("Running python script: {}", script_path);
                                              
                                              output = std::process::Command::new("python3")
                                                .args(&[script_path, &cif_gz_url, &cif_file])
                                                .output();
                                                
                                              if output.as_ref().map(|o| o.status.success()).unwrap_or(false) {
                                                  final_path = cif_file.clone();
                                                  success = true;
                                              }
                                          } else {
                                              // Curl Success -> Decompress
                                              println!("Curl success. Decompressing {}...", cif_gz_file);
                                              let _ = std::process::Command::new("gzip")
                                                .args(&["-d", "-f", &cif_gz_file])
                                                .status();
                                                
                                              final_path = cif_file.clone();
                                              success = true;
                                          }
                                     }

                                     match output {
                                         Ok(out) => {
                                             if out.status.success() {
                                                 if !final_path.ends_with(".cif") { 
                                                     final_path = output_file; 
                                                     success = true;
                                                 }
                                                 // if CIF, final_path is already set above
                                             } else {
                                                 println!("Curl Failed: {}", String::from_utf8_lossy(&out.stderr));
                                                 success = false;
                                             }
                                         },
                                         Err(e) => {
                                             println!("Failed to execute curl: {}", e);
                                             success = false;
                                         }
                                     }

                                 // 3. Fallback: PubChem (Name)
                                 } else {
                                     let name = input.clone();
                                     let url = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/SDF?record_type=3d", name);
                                     let sdf_file = format!("{}.sdf", name);
                                     println!("Fetching '{}' from PubChem...", name);
                                     
                                     let output = std::process::Command::new("curl")
                                        .args(&["-L", "-f", "-o", &sdf_file, &url])
                                        .output();
                                     
                                     match output {
                                         Ok(out) => {
                                             if out.status.success() {
                                                 final_path = sdf_file;
                                             } else {
                                                 println!("Curl Failed: {}", String::from_utf8_lossy(&out.stderr));
                                                 success = false;
                                             }
                                         },
                                         Err(e) => {
                                             println!("Failed to execute curl: {}", e);
                                             success = false;
                                         }
                                     }
                                 }

                                 // 4. Load System
                                 if success {
                                     let path_obj = std::path::Path::new(&final_path);
                                     println!("Attempting to load system from: {:?}", path_obj.canonicalize());
                                     if let Ok(meta) = std::fs::metadata(path_obj) {
                                         println!("File exists, size: {} bytes", meta.len());
                                     } else {
                                         println!("File DOES NOT EXIST at {:?}", path_obj);
                                     }
                                     
                                     match atom_core::System::load_from_file(path_obj) {
                                         Ok(new_system) => {
                                             let _ = tx.send(Ok(new_system));
                                         },
                                         Err(e) => {
                                             println!("atom_core::load_from_file FAILED: {:?}", e);
                                             let _ = tx.send(Err(format!("Failed to parse file: {}", e)));
                                         }
                                     }
                                 } else {
                                      let _ = tx.send(Err(format!("Could not find/fetch '{}'", input)));
                                 }
                             });
                         }
                    }
                }
            });

            ui.separator();
            ui.heading("Trajectory");
            if ui.button("📂 Load output.dcd").clicked() {
                if let Ok(file) = std::fs::File::open("output.dcd") {
                    if let Ok((header, frames)) = atom_core::dcd::read_dcd(file) {
                         println!("Loaded DCD: {} frames, {} atoms", header.n_frames, header.n_atoms);
                         if header.n_atoms == state.system.positions.len() as u32 {
                             state.trajectory_frames = frames;
                             state.current_frame = 0;
                             state.is_playing_dcd = true;
                             state.is_paused = true; // Pause physics
                         } else {
                             eprintln!("Atom count mismatch!");
                         }
                    }
                } else {
                    eprintln!("Could not open output.dcd (Record something first!)");
                }
            }
            
            if !state.trajectory_frames.is_empty() {
                ui.label(format!("Frame: {} / {}", state.current_frame, state.trajectory_frames.len()));
                ui.add(egui::Slider::new(&mut state.current_frame, 0..=state.trajectory_frames.len()-1).text("Timeline"));
                
                if ui.button(if state.is_playing_dcd { "⏸ Pause Replay" } else { "▶ Play Replay" }).clicked() {
                    state.is_playing_dcd = !state.is_playing_dcd;
                     if state.is_playing_dcd { state.is_paused = true; } // Ensure physics paused
                }
            } else {
                // Recording Controls (Only if not playing back)
                let rec_text = if state.is_recording { "🔴 Stop Recording" } else { "⚫ Start Recording" };
                if ui.button(rec_text).clicked() {
                    if state.is_recording {
                         state.is_recording = false;
                         state.dcd_writer = None; // Close file
                         println!("Recording stopped. Saved to output.dcd");
                    } else {
                         state.is_recording = true;
                    }
                }
            }
        });
}
