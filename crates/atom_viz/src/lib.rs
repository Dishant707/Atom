mod camera;
mod spline;
mod mesh;
pub mod ui;

use winit::{
    event::*,
    event_loop::EventLoop,
    window::{WindowBuilder, Window},
    keyboard::{KeyCode, PhysicalKey},
};
use wgpu::util::DeviceExt;
use crate::camera::{Camera, CameraController};
use atom_core::{System, SpatialGrid};
use atom_physics::{VelocityVerlet, ForceField, BerendsenThermostat};
use glam::Vec3;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// Rendering Modes
#[derive(Debug, Clone, Copy, PartialEq)]
enum RenderMode {
    Stick,      // Small spheres + bonds
    SpaceFill,  // Van der Waals radii
    Wireframe,  // Tiny spheres, bonds only
    BackboneTrace,
    Ribbon,
    Surface,
}

// Color Schemes
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorScheme {
    Element,      // CPK coloring
    Chain,        // Color by chain ID
    ResidueType,  // Hydrophobic/Polar/Charged (Basic)
    SecondaryStructure, // Helix/Sheet/Loop
    Hydrophobicity, // Kyte-Doolittle Scale
}

// Helper: Color by Hydrophobicity (Kyte-Doolittle)
// Range -4.5 (Arg) to 4.5 (Ile)
// Gradient: Blue (Hydrophilic) -> White (Neutral) -> Red (Hydrophobic)
fn color_by_hydrophobicity(residue: &str) -> glam::Vec3 {
    if let Some(h) = atom_core::get_hydrophobicity(residue) {
        // Normalize -4.5 to 4.5 -> 0.0 to 1.0
        let t = (h + 4.5) / 9.0;
        let t = t.clamp(0.0, 1.0);
        
        // Blue (0,0,1) -> White (1,1,1) -> Red (1,0,0)
        if t < 0.5 {
            // Blue to White
            let blend = t * 2.0; // 0.0 to 1.0
            glam::Vec3::new(blend, blend, 1.0)
        } else {
            // White to Red
            let blend = (t - 0.5) * 2.0; // 0.0 to 1.0
            glam::Vec3::new(1.0, 1.0 - blend, 1.0 - blend)
        }
    } else {
        glam::Vec3::new(0.5, 0.5, 0.5) // Grey for unknown (e.g. H2O, Ligands)
    }
}

// Helper: Get element name from atomic number
fn element_name(z: u8) -> &'static str {
    match z {
        1 => "Hydrogen", 6 => "Carbon", 7 => "Nitrogen", 8 => "Oxygen",
        15 => "Phosphorus", 16 => "Sulfur", 26 => "Iron", 12 => "Magnesium",
        _ => "Unknown"
    }
}

// Helper: Van der Waals radii (Angstroms)
fn vdw_radius(z: u8) -> f32 {
    match z {
        1 => 1.2,   // H
        6 => 1.7,   // C
        7 => 1.55,  // N
        8 => 1.52,  // O
        15 => 1.8,  // P
        16 => 1.8,  // S
        _ => 1.5,   // Default
    }
}

// Helper: Color by chain (hash-based)
fn color_by_chain(chain_id: &str) -> Vec3 {
    let mut hasher = DefaultHasher::new();
    chain_id.hash(&mut hasher);
    let hash = hasher.finish();
    let hue = (hash % 360) as f32;
    hsl_to_rgb(hue, 0.7, 0.6)
}

// Helper: Color by residue type (biochemistry)
fn color_by_residue(res_name: &str) -> Vec3 {
    match res_name {
        // Hydrophobic (Orange)
        "ALA" | "VAL" | "ILE" | "LEU" | "MET" | "PHE" | "TRP" | "PRO" => Vec3::new(1.0, 0.6, 0.2),
        // Polar (Cyan)
        "SER" | "THR" | "CYS" | "TYR" | "ASN" | "GLN" => Vec3::new(0.2, 0.8, 0.8),
        // Positive (Blue)
        "LYS" | "ARG" | "HIS" => Vec3::new(0.2, 0.2, 1.0),
        // Negative (Red)
        "ASP" | "GLU" => Vec3::new(1.0, 0.2, 0.2),
        // Glycine (White)
        "GLY" => Vec3::new(0.9, 0.9, 0.9),
        // Other/Ligand (Magenta)
        _ => Vec3::new(0.8, 0.2, 0.8),
    }
}

// Helper: HSL to RGB conversion
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Vec3::new(r + m, g + m, b + m)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 3],
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

struct Instance {
    position: glam::Vec3,
    color: glam::Vec3,
    scale: f32,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (glam::Mat4::from_translation(self.position) * glam::Mat4::from_scale(glam::Vec3::splat(self.scale))).to_cols_array_2d(),
            color: self.color.to_array(),
        }
    }
}

fn create_sphere_mesh(radius: f32, lat_segments: u32, long_segments: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for lat in 0..=lat_segments {
        let theta = lat as f32 * std::f32::consts::PI / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=long_segments {
            let phi = lon as f32 * 2.0 * std::f32::consts::PI / long_segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = cos_phi * sin_theta;
            let y = cos_theta;
            let z = sin_phi * sin_theta;

            // Normal is just the direction (x,y,z) for unit sphere logic (before value scaling)
            // Or just normalized position relative to center.
            let nx = x;
            let ny = y;
            let nz = z;

            vertices.push(Vertex {
                position: [x * radius, y * radius, z * radius],
                normal: [nx, ny, nz],
            });
        }
    }

    for lat in 0..lat_segments {
        for lon in 0..long_segments {
            let first = (lat * (long_segments + 1)) + lon;
            let second = first + long_segments + 1;

            indices.push(first as u16);
            indices.push(second as u16);
            indices.push((first + 1) as u16);

            indices.push(second as u16);
            indices.push((second + 1) as u16);
            indices.push((first + 1) as u16);
        }
    }

    (vertices, indices)
}

fn create_cylinder_mesh(radius: f32, length: f32, segments: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Top and Bottom centers
    let top_y = length / 2.0;
    let bottom_y = -length / 2.0;

    // Side vertices
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let (sin, cos) = theta.sin_cos();
        let x = cos * radius;
        let z = sin * radius;
        
        // Normal for side is just (x, 0, z) normalized (which is cos, 0, sin)
        let nx = cos;
        let nz = sin;

        // Top ring
        vertices.push(Vertex {
            position: [x, top_y, z],
            normal: [nx, 0.0, nz],
        });
        
        // Bottom ring
        vertices.push(Vertex {
            position: [x, bottom_y, z],
            normal: [nx, 0.0, nz],
        });
    }

    // Indices for sides
    for i in 0..segments {
        let top1 = i * 2;
        let bottom1 = i * 2 + 1;
        let top2 = i * 2 + 2 ;
        let bottom2 = i * 2 + 3 ;

        // Quad formed by two triangles
        indices.push(top1 as u16);
        indices.push(bottom1 as u16);
        indices.push(top2 as u16);

        indices.push(bottom1 as u16);
        indices.push(bottom2 as u16);
        indices.push(top2 as u16);
    }

    (vertices, indices)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl LineVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    window: &'a Window,
    
    // Sphere Renderer
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,

    // Bond Renderer (Cylinders)
    // Reuses render_pipeline (same vertex format)
    cylinder_vertex_buffer: wgpu::Buffer,
    cylinder_index_buffer: wgpu::Buffer,
    num_cylinder_indices: u32,
    bond_instance_buffer: wgpu::Buffer,
    num_bonds: u32,

    pub(crate) camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    
    // Depth Buffer
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    
    // Physics State
    pub(crate) system: System,
    grid: SpatialGrid,
    integrator: VelocityVerlet,
    force_field: Box<dyn ForceField>,
    thermostat: BerendsenThermostat,
    
    // Simulation Control
    pub(crate) is_paused: bool,
    
    // Visualization Settings
    pub(crate) atom_scale: f32,
    pub(crate) bond_radius: f32,
    pub(crate) show_forces: bool,
    force_scale: f32,
    
    // Force Renderer (Cylinders)
    force_instance_buffer: wgpu::Buffer,
    num_forces: u32,

    // Ribbon Renderer
    ribbon_pipeline: wgpu::RenderPipeline,
    ribbon_vertex_buffer: wgpu::Buffer,
    ribbon_index_buffer: wgpu::Buffer,
    num_ribbon_indices: u32,

    // Surface Renderer
    surface_pipeline: wgpu::RenderPipeline,
    surface_vertex_buffer: Option<wgpu::Buffer>,
    surface_index_buffer: Option<wgpu::Buffer>,
    num_surface_indices: u32,

    pub custom_colors: std::collections::HashMap<usize, [f32; 3]>,

    // Trajectory Recording
    pub(crate) dcd_writer: Option<atom_core::dcd::DCDWriter<std::fs::File>>,
    pub(crate) is_recording: bool,
    
    // Trajectory Playback
    pub(crate) trajectory_frames: Vec<Vec<Vec3>>,
    pub(crate) current_frame: usize,
    pub(crate) is_playing_dcd: bool,

    // State to track mouse for camera
    last_mouse_pos: Option<(f64, f64)>,
    mouse_delta: Option<(f64, f64)>,
    scroll_delta: Option<f32>,
    
    // UI State
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    
    // New Visualization Features
    pub(crate) render_mode: RenderMode,
    pub(crate) color_scheme: ColorScheme,
    pub(crate) is_auto_rotating: bool,
    pub(crate) rotation_speed: f32,
    pub(crate) selected_atom: Option<usize>,
    pub(crate) selected_component: std::collections::HashSet<usize>,
    pub(crate) is_isolated: bool,
    pub(crate) should_rebuild_buffers: bool,
    pub(crate) needs_buffer_update: bool,
    
    // UI Loading State
    pub(crate) load_input: String,
    
    // Measurement Tool
    pub(crate) measurement_mode: bool,
    pub(crate) measurement_click_buffer: Vec<usize>, // Stores 1st click
    pub(crate) measurements: Vec<(usize, usize, f32)>, // A, B, Distance

    // Sequence Viewer
    pub(crate) residue_cache: Vec<ResidueInfo>,
    
    // Async Loading
    pub(crate) load_rx: Option<std::sync::mpsc::Receiver<Result<System, String>>>,
    pub(crate) is_loading: bool,
}

#[derive(Debug, Clone)]
pub struct ResidueInfo {
    pub name: String,
    pub id: u32,
    pub chain: String,
    pub atom_range: std::ops::Range<usize>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_pos: [f32; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use glam::Mat4;
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view_pos: [0.0; 4],
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().to_cols_array_2d();
        let pos = camera.eye;
        self.view_pos = [pos.x, pos.y, pos.z, 1.0];
    }
}

impl<'a> State<'a> {
    async fn new(window: &'a Window, initial_system: System, custom_force_field: Option<Box<dyn ForceField>>, custom_colors: std::collections::HashMap<usize, [f32; 3]>) -> State<'a> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
            
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // System setup
        let system = initial_system;
        let grid = SpatialGrid::new(system.box_size, 2.5);
        let integrator = VelocityVerlet::new(0.016);
        // Smart Force Field Selection
        let force_field: Box<dyn ForceField> = if let Some(ffield) = custom_force_field {
             ffield
        } else {
            // Use Molecular Force Field (Bonds + VDW) to prevent collapse
            Box::new(atom_physics::MolecularForceField::new())
        };
        let thermostat = BerendsenThermostat::new(1.0, 10.0); // Target T=1.0, tau=10.0

        // Camera
        let box_center = system.box_size * 0.5;
        let camera = Camera {
            eye: (box_center.x, box_center.y, box_center.z + 40.0).into(),
            target: box_center,
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let camera_controller = CameraController::new(0.5);

        // Depth Texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Shader
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Build Residue Cache for Sequence Viewer
        let mut residue_cache = Vec::new();
        if !system.chain_ids.is_empty() && !system.residue_ids.is_empty() {
            let mut start_idx = 0;
            let mut last_id = if !system.residue_ids.is_empty() { system.residue_ids[0] } else { 0 };
            let mut last_chain = if !system.chain_ids.is_empty() { system.chain_ids[0].clone() } else { "?".to_string() };
            // Optional: last_name to detect rename? usually ID change is enough.
            let mut last_name = if !system.residue_names.is_empty() { system.residue_names[0].clone() } else { "UNK".to_string() };

            for i in 1..system.positions.len() {
                let mut changed = false;
                // Check Bounds
                let curr_id = if i < system.residue_ids.len() { system.residue_ids[i] } else { 0 };
                let curr_chain = if i < system.chain_ids.len() { &system.chain_ids[i] } else { "?" };
                
                if curr_id != last_id || curr_chain != last_chain {
                    changed = true;
                }
                
                if changed {
                    // Push previous
                    residue_cache.push(ResidueInfo {
                        name: last_name.clone(),
                        id: last_id,
                        chain: last_chain.clone(),
                        atom_range: start_idx..i,
                    });
                    
                    // Reset
                    start_idx = i;
                    last_id = curr_id;
                    last_chain = curr_chain.to_string();
                    last_name = if i < system.residue_names.len() { system.residue_names[i].clone() } else { "UNK".to_string() };
                }
            }
            // Push final
            if start_idx < system.positions.len() {
                residue_cache.push(ResidueInfo {
                    name: last_name,
                    id: last_id,
                    chain: last_chain,
                    atom_range: start_idx..system.positions.len(),
                });
            }
        }
        println!("Built Residue Cache: {} residues", residue_cache.len());

        // Sphere Geometry - Level of Detail (LOD)
        let atom_count = system.positions.len();
        let (sphere_lat, sphere_lon) = if atom_count > 500_000 {
            (4, 4) // Low Poly (Massive)
        } else if atom_count > 100_000 {
            (8, 8) // Medium Poly (Large)
        } else {
            (16, 16) // High Poly (Standard)
        };
        println!("Initializing optimized geometry: {} atoms -> {}x{} sphere segments", atom_count, sphere_lat, sphere_lon);
        let (vertices, indices) = create_sphere_mesh(1.0, sphere_lat, sphere_lon); 
        
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            }
        );
        let num_indices = indices.len() as u32;

        // Instances
        let instances = system.positions.iter().enumerate().map(|(i, &pos)| {
            let atomic_number = if i < system.atomic_numbers.len() { system.atomic_numbers[i] } else { 6 };
            
            // CPK Coloring
            let color = if let Some(c) = custom_colors.get(&i) {
                glam::Vec3::from(*c)
            } else {
                match atomic_number {
                    1 => glam::Vec3::new(1.0, 1.0, 1.0), // H
                    6 => glam::Vec3::new(0.2, 0.2, 0.2), // C
                    7 => glam::Vec3::new(0.0, 0.0, 1.0), // N
                    8 => glam::Vec3::new(1.0, 0.0, 0.0), // O
                    16 => glam::Vec3::new(1.0, 1.0, 0.0), // S
                    15 => glam::Vec3::new(1.0, 0.65, 0.0), // P
                    _ => glam::Vec3::new(0.8, 0.0, 0.8), // Magenta
                }
            };
            
            // Scale based on Atomic Number (VDW Radius approx) + Global Scale for visibility
            // User requested "bubbles" -> larger atoms. 
            let radius = match atomic_number {
                 1 => 1.2,
                 6 => 1.7,
                 7 => 1.55,
                 8 => 1.52,
                 15 => 1.8,
                 16 => 1.8,
                 _ => 1.7,
            };
            
            // "More clearly visible" -> Scale multiplier
            // User requested SMALLER atoms to see bonds and reduce computation/overdraw.
            // Ball-and-Stick style: radius * 0.4
            let scale = radius * 0.4; 
            
            Instance { position: pos, color, scale }
        }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        let num_instances = instance_data.len() as u32;

        // Ribbon Pipeline
        let ribbon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ribbon Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_ribbon",
                buffers: &[mesh::RibbonVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let ribbon_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ribbon Vertex Buffer"),
            size: 1024 * 1024 * 10, // 10MB initial
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let ribbon_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ribbon Index Buffer"),
            size: 1024 * 1024 * 4, // 4MB initial
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let num_ribbon_indices = 0;

        // Surface Pipeline
        let surface_shader = device.create_shader_module(wgpu::include_wgsl!("shader_surface.wgsl"));
        
        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &surface_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &surface_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Bond Geometry (Standard Cylinder: Radius 0.1, Length 1.0 aligned Y)
        let cyl_segments = if atom_count > 50_000 {
            3
        } else if atom_count > 10_000 {
            4
        } else {
            8
        };
        let (cyl_verts, cyl_inds) = create_cylinder_mesh(0.1, 1.0, cyl_segments);
        
        let cylinder_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Cylinder Vertex Buffer"),
                contents: bytemuck::cast_slice(&cyl_verts),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let cylinder_index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Cylinder Index Buffer"),
                contents: bytemuck::cast_slice(&cyl_inds),
                usage: wgpu::BufferUsages::INDEX,
            }
        );
        let num_cylinder_indices = cyl_inds.len() as u32;
        
        // Force Instance Buffer (Max 1 force per atom)
        let max_forces = system.positions.len() as u64;
        let force_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Force Instance Buffer"),
            size: max_forces * std::mem::size_of::<InstanceRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_bonds = 100_000;
        let bond_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bond Instance Buffer"),
            size: (max_bonds * std::mem::size_of::<InstanceRaw>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Egui Setup
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            instance_buffer,
            num_instances,
            
            cylinder_vertex_buffer,
            cylinder_index_buffer,
            num_cylinder_indices,
            bond_instance_buffer,
            num_bonds: 0,
            
            force_instance_buffer,
            num_forces: 0,

            ribbon_pipeline,
            ribbon_vertex_buffer,
            ribbon_index_buffer,

            num_ribbon_indices,

            surface_pipeline,
            surface_vertex_buffer: None,
            surface_index_buffer: None,
            num_surface_indices: 0,
            
            custom_colors,

            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            
            // Depth
            depth_texture,
            depth_view,

            system,
            grid,
            integrator,
            force_field,
            thermostat,
            is_paused: true, // Start paused to allow viewing without physics instability
            
            atom_scale: 1.0,
            bond_radius: 0.1,
            show_forces: false,
            force_scale: 0.1, // Forces can be large/small, need scaling
            
            dcd_writer: None,
            is_recording: false,
            
            trajectory_frames: Vec::new(),
            current_frame: 0,
            is_playing_dcd: false,

            egui_ctx,
            egui_state,
            egui_renderer,
            
            // New Visualization Features
            render_mode: RenderMode::Ribbon,
            color_scheme: ColorScheme::Chain, // Color by chain to see issues better
            is_auto_rotating: false,
            rotation_speed: 10.0,
            selected_atom: None,
            selected_component: std::collections::HashSet::new(),
            is_isolated: false,
            should_rebuild_buffers: true,
            needs_buffer_update: true,
            residue_cache,
            load_rx: None,
            is_loading: false,
            
            last_mouse_pos: None,
            mouse_delta: None,
            scroll_delta: None,
            
            // UI Loading
            load_input: String::new(),
            
            measurement_mode: false,
            measurement_click_buffer: Vec::new(),
            measurements: Vec::new(),
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            
            // Resize Depth Texture
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        // Egui Input
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.consumed {
            return true;
        }

        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Space),
                    ..
                },
                ..
            } => {
                self.is_paused = !self.is_paused;
                println!("Simulation Paused: {}", self.is_paused);
                true
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyR),
                    ..
                },
                ..
            } => {
                self.render_mode = match self.render_mode {
                    RenderMode::Stick => RenderMode::SpaceFill,
                    RenderMode::SpaceFill => RenderMode::Wireframe,
                    RenderMode::Wireframe => RenderMode::BackboneTrace,
                    RenderMode::BackboneTrace => RenderMode::Ribbon,
                    RenderMode::Ribbon => RenderMode::Surface,
                    RenderMode::Surface => RenderMode::Stick,
                };
                println!("Render Mode: {:?}", self.render_mode);
                self.needs_buffer_update = true; // Refresh static scene
                true
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyC),
                    ..
                },
                ..
            } => {
                self.color_scheme = match self.color_scheme {
                    ColorScheme::Element => ColorScheme::Chain,
                    ColorScheme::Chain => ColorScheme::ResidueType,
                    ColorScheme::ResidueType => ColorScheme::SecondaryStructure,
                    ColorScheme::SecondaryStructure => ColorScheme::Hydrophobicity,
                    ColorScheme::Hydrophobicity => ColorScheme::Element,
                };
                println!("Color Scheme: {:?}", self.color_scheme);
                self.needs_buffer_update = true; // Refresh static scene
                true
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyT),
                    ..
                },
                ..
            } => {
                self.is_auto_rotating = !self.is_auto_rotating;
                println!("Auto-Rotation: {}", self.is_auto_rotating);
                true
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyS),
                    ..
                },
                ..
            } => {
                self.save_screenshot();
                true
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // Raycast Selection (if not using Shift/Alt for camera)
                if !self.camera_controller.is_shift_pressed && !self.camera_controller.is_alt_pressed {
                     if let Some((mx, my)) = self.last_mouse_pos {
                        let width = self.config.width as f32;
                        let height = self.config.height as f32;
                        let ray = self.camera.screen_point_to_ray(mx as f32, my as f32, width, height);
                        self.select_atom_from_ray(ray);
                     }
                }
                
                self.camera_controller.process_events(event);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let this_pos = (position.x, position.y);
                if let Some(last) = self.last_mouse_pos {
                    self.mouse_delta = Some((this_pos.0 - last.0, this_pos.1 - last.1));
                }
                self.last_mouse_pos = Some(this_pos);
                // Also let camera controller see generic events if needed
                self.camera_controller.process_events(event);
                true
            }
            // Reset mouse state if cursor leaves window or focus is lost
            WindowEvent::CursorLeft { .. } | WindowEvent::Focused(false) => {
                self.last_mouse_pos = None;
                self.mouse_delta = None;
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.scroll_delta = Some(*y);
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                         self.scroll_delta = Some(pos.y as f32 * 0.01);
                    }
                }
                true
            }
            // Clear delta on input end? No, clear in update.
            _ => self.camera_controller.process_events(event),
        }
    }

    fn update(&mut self) {
        if self.should_rebuild_buffers {
            self.rebuild_buffers();
            self.should_rebuild_buffers = false;
            // Also reset camera to look at new molecule
            let box_center = self.system.box_size * 0.5;
            self.camera.target = box_center;
            self.camera.eye = glam::Vec3::new(box_center.x, box_center.y, box_center.z + 40.0);
            self.is_paused = true; // Auto-pause on load for safety
        }

        // Check for Async Load Results
        if let Some(rx) = &self.load_rx {
             if let Ok(result) = rx.try_recv() {
                 self.is_loading = false;
                 match result {
                     Ok(new_system) => {
                         println!("Async Load Complete: {} atoms", new_system.positions.len());
                         self.system = new_system;
                         
                         // Reset Camera
                         let center = self.system.box_size * 0.5;
                         self.camera.target = center;
                         self.camera.eye = glam::Vec3::new(center.x, center.y, center.z + 40.0);
                         
                         // Re-Evaluate LOD
                         let atom_count = self.system.positions.len();
                         let (sphere_lat, sphere_lon) = if atom_count > 500_000 {
                            (4, 4) 
                        } else if atom_count > 100_000 {
                            (8, 8)
                        } else {
                            (16, 16)
                        };
                        println!("LOD Update: {}x{}", sphere_lat, sphere_lon);
                        let (vertices, indices) = create_sphere_mesh(1.0, sphere_lat, sphere_lon);
                        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Sphere Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Sphere Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                        self.num_indices = indices.len() as u32;

                         self.should_rebuild_buffers = true;
                         self.needs_buffer_update = true;
                         self.load_rx = None; // clear channel
                     },
                     Err(e) => {
                         println!("Async Load Failed: {}", e);
                         self.load_rx = None;
                     }
                 }
             }
        }

        // Pass Deltas to Camera
        // If deltas are NaN (from bad resize/jump), filter them
        if let Some((dx, dy)) = self.mouse_delta {
             if dx.is_finite() && dy.is_finite() {
                 self.camera_controller.update_camera(&mut self.camera, self.mouse_delta, self.scroll_delta);
             }
        } else {
             self.camera_controller.update_camera(&mut self.camera, None, self.scroll_delta);
        }
        
        // Reset Deltas
        self.mouse_delta = None;
        self.scroll_delta = None;
        
        // Auto-Rotation
        if self.is_auto_rotating {
            let angle = self.rotation_speed * 0.01; // Radians per frame
            let rotation = glam::Quat::from_axis_angle(glam::Vec3::Y, angle);
            let offset = self.camera.eye - self.camera.target;
            self.camera.eye = self.camera.target + rotation * offset;
        }
        
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

        if self.is_playing_dcd && !self.trajectory_frames.is_empty() {
             // Playback Mode
             self.current_frame = (self.current_frame + 1) % self.trajectory_frames.len();
             self.system.positions = self.trajectory_frames[self.current_frame].clone();
             
             // Optional: Re-calculate grid/bonds if dynamic? 
             // Bonds are index-based so they persist. 
        } else if !self.is_paused {
             // Physics Logic
             // ANI forces are STRONG - use very small timesteps
             // Reduce substeps for performance (ANI is expensive!)
             let sub_steps = 2; // Reduced from 10
             let sub_dt = 0.0005; // Very small timestep - 0.5 femtoseconds
             
             let old_dt = self.integrator.dt;
             self.integrator.dt = sub_dt;
     
             for _ in 0..sub_steps {
                 // Update spatial grid
                 self.grid.insert(&self.system);
                 // Run physics integration with ANI force field
                 self.integrator.integrate(&mut self.system, &self.grid, &mut *self.force_field);
                 // Apply thermostat to maintain 300K
                 self.thermostat.apply(&mut self.system, 300.0);
                 
                 // Safety: Check for position explosion (NaN or too far from center)
                 let center = self.system.box_size * 0.5;
                 let max_dist = self.system.box_size.length() * 2.0;
                 for (i, pos) in self.system.positions.iter_mut().enumerate() {
                     if pos.is_nan() {
                         eprintln!("⚠️ Atom {} Position is NaN! Forces: {:?}", i, self.system.forces.get(i));
                         *pos = center;
                     } else if (*pos - center).length() > max_dist {
                         eprintln!("⚠️ Atom {} Exploded! Dist: {:.2}, Pos: {:?}", i, (*pos - center).length(), *pos);
                         *pos = center;
                     }
                 }
             }
             
             self.integrator.dt = old_dt; // Restore if needed, though we just run loop
            
            // Recording Logic
            if self.is_recording {
                if self.dcd_writer.is_none() {
                    println!("Starting Recording to output.dcd...");
                    let file = std::fs::File::create("output.dcd").unwrap();
                    let writer = atom_core::dcd::DCDWriter::new(file, self.system.positions.len() as u32).unwrap();
                    self.dcd_writer = Some(writer);
                }
                
                if let Some(writer) = &mut self.dcd_writer {
                    if let Err(e) = writer.write_frame(&self.system.positions) {
                        eprintln!("Failed to write DCD frame: {}", e);
                        self.is_recording = false; // Stop recording on error
                    }
                }
            }
        }


        if self.is_paused && !self.is_playing_dcd && !self.needs_buffer_update {
            return;
        }

        self.needs_buffer_update = false;

        // Update Instance Buffer
        let instance_data = self.system.positions.iter().enumerate().map(|(i, &pos)| {
             // Isolation Logic
             if self.is_isolated {
                 // Check if i is in selected_component
                 // Note: Contains is O(n), but for viz update (once per frame or click) on small lists it's ok. 
                 // For large systems, better to use HashSet but Vec is fine for now (ligands are small).
                 // Actually, "Connected Component" can be large.
                 // Optimization: Only render if valid.
                 if !self.selected_component.contains(&i) {
                     // Check if it's the selected atom itself just in case
                     if Some(i) != self.selected_atom {
                         return InstanceRaw {
                             model: glam::Mat4::ZERO.to_cols_array_2d(), // Invisible
                             color: [0.0; 3],
                         };
                     }
                 }
             }

             let atomic_number = if i < self.system.atomic_numbers.len() { 
                 self.system.atomic_numbers[i] 
             } else { 
                 6 
             };
            
            // Determine atom scale based on render mode
            let scale = match self.render_mode {
                RenderMode::Stick => self.atom_scale * 0.8, // Slightly smaller sticks
                RenderMode::SpaceFill => vdw_radius(atomic_number) * 0.7, // 70% VDW for clearer separation
                RenderMode::Wireframe => self.atom_scale * 0.25,
                RenderMode::BackboneTrace => {
                    if i < self.system.atom_names.len() && self.system.atom_names[i] == "CA" {
                        self.atom_scale * 0.4 // Visible CA atoms
                    } else {
                        0.0 // Hide other atoms
                    }
                },
                RenderMode::Ribbon => {
                    let mut is_visible = false;
                    // Hybrid Rendering: Show Ligands/Ions as SpaceFill
                    // Hide Standard AA and Water
                    if i < self.system.residue_names.len() {
                        let res = &self.system.residue_names[i];
                        let std_aa = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"];
                        let nucleotides = ["DA", "DT", "DC", "DG", "DI", "A", "U", "C", "G"];
                        let solvents = ["HOH", "WAT", "SOL"];
                        
                        // Show everything that is NOT protein, DNA/RNA, or Water
                        if !std_aa.contains(&res.as_str()) && !nucleotides.contains(&res.as_str()) && !solvents.contains(&res.as_str()) {
                            is_visible = true;
                        }
                    }
                    
                    if is_visible {
                        vdw_radius(atomic_number) * 0.6 
                    } else {
                        0.0 
                    }
                },
                RenderMode::Surface => 0.0,
            };
            let atom_scale_vec = glam::Vec3::splat(scale);
            
            // Determine color based on color scheme
            let color = match self.color_scheme {
                ColorScheme::Element => {
                    // CPK Coloring
                    match atomic_number {
                        1 => glam::Vec3::new(1.0, 1.0, 1.0), 
                        6 => glam::Vec3::new(0.2, 0.2, 0.2), 
                        7 => glam::Vec3::new(0.0, 0.0, 1.0), 
                        8 => glam::Vec3::new(1.0, 0.0, 0.0), 
                        16 => glam::Vec3::new(1.0, 1.0, 0.0), 
                        15 => glam::Vec3::new(1.0, 0.65, 0.0), 
                        _ => glam::Vec3::new(0.8, 0.0, 0.8), 
                    }
                },
                ColorScheme::Chain => {
                    if i < self.system.chain_ids.len() {
                        color_by_chain(&self.system.chain_ids[i])
                    } else {
                        glam::Vec3::new(0.5, 0.5, 0.5)
                    }
                },
                ColorScheme::ResidueType => {
                    if i < self.system.residue_names.len() {
                        color_by_residue(&self.system.residue_names[i])
                    } else {
                        glam::Vec3::new(0.5, 0.5, 0.5)
                    }
                },
                ColorScheme::SecondaryStructure => {
                    use atom_core::SecondaryStructure;
                    if i < self.system.secondary_structure.len() {
                        match self.system.secondary_structure[i] {
                            SecondaryStructure::Helix => glam::Vec3::new(1.0, 0.0, 1.0), // Magenta
                            SecondaryStructure::Sheet => glam::Vec3::new(1.0, 1.0, 0.0), // Yellow
                            SecondaryStructure::Loop => glam::Vec3::new(0.9, 0.9, 0.9),  // White/Light Gray
                            SecondaryStructure::Unknown => glam::Vec3::new(0.5, 0.5, 0.5), // Gray
                        }
                    } else {
                        glam::Vec3::new(0.5, 0.5, 0.5)
                    }
                },
                ColorScheme::Hydrophobicity => if i < self.system.residue_names.len() { color_by_hydrophobicity(&self.system.residue_names[i]) } else { glam::Vec3::new(0.5, 0.5, 0.5) },
            };

            InstanceRaw {
                // Apply Scale * Translation
                model: glam::Mat4::from_scale_rotation_translation(atom_scale_vec, glam::Quat::IDENTITY, pos).to_cols_array_2d(),
                color: color.to_array(),
            }
        }).collect::<Vec<_>>();
        
        self.instance_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        self.num_instances = instance_data.len() as u32;

        // Generate Surface (Expensive!)
        if self.render_mode == RenderMode::Surface && !self.system.positions.is_empty() {
             let probe_radius = 1.4;
             let grid_res = 1.0; // Moderate Quality
             println!("Generating Surface for {} atoms...", self.system.positions.len());
             let surface_mesh = atom_core::surface::generate_surface(&self.system, probe_radius, grid_res);
             
             let vertices = surface_mesh.vertices.iter().zip(surface_mesh.normals.iter()).map(|(p, n)| {
                 Vertex { position: p.to_array(), normal: n.to_array() }
             }).collect::<Vec<_>>();
             
             self.surface_vertex_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                 label: Some("Surface Vertex Buffer"),
                 contents: bytemuck::cast_slice(&vertices),
                 usage: wgpu::BufferUsages::VERTEX,
             }));
             
             self.surface_index_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                 label: Some("Surface Index Buffer"),
                 contents: bytemuck::cast_slice(&surface_mesh.indices),
                 usage: wgpu::BufferUsages::INDEX,
             }));
             self.num_surface_indices = surface_mesh.indices.len() as u32;
        }
        
        // Update Bond Instances (Sticks)
        if self.render_mode == RenderMode::BackboneTrace {
             // Trace Logic: Connect CA atoms
             let mut trace_instances = Vec::new();
             let trace_radius = 0.2; // Thicker than bonds
             
             // Collect CA atoms indices
             let mut ca_indices = Vec::new();
             for (i, name) in self.system.atom_names.iter().enumerate() {
                 if name == "CA" {
                     ca_indices.push(i);
                 }
             }
             
             // Connect sequential CA atoms
             for k in 0..ca_indices.len().saturating_sub(1) {
                 let i = ca_indices[k];
                 let j = ca_indices[k+1];
                 
                 // Check if same chain and sequential residue
                 let chain_i = &self.system.chain_ids[i];
                 let chain_j = &self.system.chain_ids[j];
                 let res_i = self.system.residue_ids[i];
                 let res_j = self.system.residue_ids[j];
                 
                 // Allow max gap of 1 (e.g. 10 -> 11). If gap > 1, assume break.
                 if chain_i == chain_j && (res_j as i32 - res_i as i32).abs() <= 1 {
                     let start = self.system.positions[i];
                     let end = self.system.positions[j];
                     
                     let diff = end - start;
                     let length = diff.length();
                     let center = (start + end) * 0.5;
                     
                     // Use color of start atom (simplified)
                     // Re-calculate color based on scheme for atom i
                     let color = match self.color_scheme {
                        ColorScheme::Element => glam::Vec3::new(0.5, 0.5, 0.5), // Uniform gray for trace
                        ColorScheme::Chain => if i < self.system.chain_ids.len() { color_by_chain(&self.system.chain_ids[i]) } else { glam::Vec3::ZERO },
                        ColorScheme::ResidueType => if i < self.system.residue_names.len() { color_by_residue(&self.system.residue_names[i]) } else { glam::Vec3::ZERO },
                        ColorScheme::SecondaryStructure => {
                             if i < self.system.secondary_structure.len() {
                                 match self.system.secondary_structure[i] {
                                     atom_core::SecondaryStructure::Helix => glam::Vec3::new(1.0, 0.0, 1.0),
                                     atom_core::SecondaryStructure::Sheet => glam::Vec3::new(1.0, 1.0, 0.0),
                                     atom_core::SecondaryStructure::Loop => glam::Vec3::new(0.9, 0.9, 0.9),
                                     atom_core::SecondaryStructure::Unknown => glam::Vec3::new(0.5, 0.5, 0.5),
                                 }
                             } else { glam::Vec3::new(0.5, 0.5, 0.5) }
                        },
                        ColorScheme::Hydrophobicity => if i < self.system.residue_names.len() { color_by_hydrophobicity(&self.system.residue_names[i]) } else { glam::Vec3::new(0.5, 0.5, 0.5) },
                     };

                     let rotation = glam::Quat::from_rotation_arc(glam::Vec3::Y, diff.normalize());
                     let scale = glam::Vec3::new(trace_radius, length, trace_radius);
                     let model = glam::Mat4::from_scale_rotation_translation(scale, rotation, center);
                     
                     trace_instances.push(InstanceRaw {
                         model: model.to_cols_array_2d(),
                         color: color.to_array(),
                     });
                 }
             }
             
             self.num_bonds = trace_instances.len() as u32;
             self.bond_instance_buffer = self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Bond Instance Buffer (Trace)"),
                    contents: bytemuck::cast_slice(&trace_instances),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }
             );
         } else {
             // Standard Bond Logic (Stick/Wireframe OR Ribbon for Ligands)
             // In Ribbon mode, we want to show bonds ONLY for non-protein residues (Ligands)
             let mut bond_instances = Vec::new();
             let bond_radius_scale = self.bond_radius / 0.1;
             
             let is_ribbon = self.render_mode == RenderMode::Ribbon;
             
             for &(i, j, order) in &self.system.bonds {
                 if i < self.system.positions.len() as u32 && j < self.system.positions.len() as u32 {
                      
                      if is_ribbon {
                          // Check if both atoms are in standard AA residues -> Hide Bond
                          let mut hide = false;
                          let len = self.system.residue_names.len() as u32;
                          if i < len && j < len {
                              let r1 = &self.system.residue_names[i as usize];
                              let r2 = &self.system.residue_names[j as usize];
                              let std_aa = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "WAT", "HOH"];
                              
                              if std_aa.contains(&r1.as_str()) && std_aa.contains(&r2.as_str()) {
                                  hide = true;
                              }
                          }
                          if hide { continue; }
                      }

                      let start = self.system.positions[i as usize];
                      let end = self.system.positions[j as usize];
                      
                      let diff = end - start;
                      let length = diff.length();
                      if length < 0.1 { continue; }
                      let midpoint = start + diff * 0.5;
                      
                      let rotation = glam::Quat::from_rotation_arc(glam::Vec3::Y, diff.normalize());
                      let _radius = if self.render_mode == RenderMode::Wireframe { 0.04 } else { 0.08 };
                      
                      // Base Color (Grey)
                      let color = [0.6, 0.6, 0.6];

                      // Multi-Bond Offsets
                      let dir = diff.normalize();
                      let mut arbitrary = glam::Vec3::X;
                      if dir.dot(arbitrary).abs() > 0.9 { arbitrary = glam::Vec3::Z; }
                      let right = dir.cross(arbitrary).normalize();
                      
                      let offset_dist = 0.12; 

                      // Helper to add instance
                      let mut add_cyl = |offset: glam::Vec3| {
                           let center = midpoint + offset;
                           let scale = glam::Vec3::new(bond_radius_scale * if order == atom_core::BondOrder::Single { 1.0 } else { 0.6 }, length, bond_radius_scale * if order == atom_core::BondOrder::Single { 1.0 } else { 0.6 });
                           let model = glam::Mat4::from_scale_rotation_translation(scale, rotation, center);
                           bond_instances.push(InstanceRaw { model: model.to_cols_array_2d(), color });
                      };

                      match order {
                          atom_core::BondOrder::Double => {
                              add_cyl(right * offset_dist);
                              add_cyl(right * -offset_dist);
                          },
                          atom_core::BondOrder::Triple => {
                              add_cyl(glam::Vec3::ZERO);
                              add_cyl(right * offset_dist * 1.2);
                              add_cyl(right * -offset_dist * 1.2);
                          },
                          _ => {
                              // Standard Single Bond (Center)
                              let scale = glam::Vec3::new(bond_radius_scale, length, bond_radius_scale);
                              let model = glam::Mat4::from_scale_rotation_translation(scale, rotation, midpoint);
                              
                              // Keep Strain Color for Single Bonds? It was cool.
                              /*
                              let eq_len = match (self.system.atomic_numbers[i as usize].min(self.system.atomic_numbers[j as usize]), 
                                                self.system.atomic_numbers[i as usize].max(self.system.atomic_numbers[j as usize])) {
                                  (6, 6) => 1.54, (1, 6) => 1.09, (1, 1) => 0.74, (1, 8) => 0.96, _ => 1.5,
                              };
                              let strain = (length - eq_len) / eq_len;
                              let strain_color = if strain < -0.1 { [0.2, 0.2, 1.0, 1.0] } else if strain > 0.1 { [1.0, 0.2, 0.2, 1.0] } else { [0.6, 0.8, 0.6, 1.0] };
                              */
                              // Let's stick to Uniform Grey for consistency with multi-bonds for now.
                              
                              bond_instances.push(InstanceRaw { model: model.to_cols_array_2d(), color });
                          }
                      }
                 }
             }
             
             self.num_bonds = bond_instances.len() as u32;
             self.bond_instance_buffer = self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Bond Instance Buffer"),
                    contents: bytemuck::cast_slice(&bond_instances),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }
             );
          }

          
        if self.render_mode == RenderMode::Ribbon {
            let mut ribbon_vertices: Vec<mesh::RibbonVertex> = Vec::new();
            let mut ribbon_indices: Vec<u32> = Vec::new();

            // Group by Chain
            let mut current_chain_id = String::new();
            let mut chain_positions: Vec<glam::Vec3> = Vec::new();
            let mut chain_structures: Vec<atom_core::SecondaryStructure> = Vec::new();
            let mut chain_colors: Vec<glam::Vec3> = Vec::new();
            let mut last_residue_id = -1000;

            let ribbon_quality = if self.system.positions.len() > 50_000 { 1 } else if self.system.positions.len() > 10_000 { 2 } else { 4 };

            let flush_chain = |pos: &[glam::Vec3], str: &[atom_core::SecondaryStructure], cols: &[glam::Vec3], 
                                   out_verts: &mut Vec<mesh::RibbonVertex>, out_inds: &mut Vec<u32>| {
                let start_index = out_verts.len() as u32;
                let (verts, inds) = mesh::RibbonMeshGenerator::generate_chain_mesh(pos, str, cols, ribbon_quality); 
                out_verts.extend(verts);
                for i in inds {
                     out_inds.push(start_index + i);
                }
            };
            
            // Collect CA atoms (Protein) and C4' atoms (Nucleic Acid)
            for (i, name) in self.system.atom_names.iter().enumerate() {
                let is_protein = name == "CA";
                let is_nucleic = name == "C4'"; // C4' is a robust backbone marker for RNA/DNA

                if is_protein || is_nucleic {
                    let chain = if i < self.system.chain_ids.len() { self.system.chain_ids[i].clone() } else { "A".to_string() };
                    let atom_res_id = self.system.residue_ids[i] as i32;

                    // Handle Duplicate Residues (AltLocs): If same chain and same residue ID, skip.
                    if chain == current_chain_id && atom_res_id == last_residue_id {
                        continue; 
                    }

                    // Check for break (chain change or residue gap > 1)
                    // Check for break (chain change or residue gap > 1)
                    // Note: residue_id might wrap or be non-sequential in rare cases, but usually increasing.
                    // Important: Use i64 to avoid overflow during subtraction.
                    let gap = (atom_res_id as i64 - last_residue_id as i64).abs();
                    if chain != current_chain_id || gap > 1 {
                         // Flush previous
                         if !chain_positions.is_empty() {
                             flush_chain(&chain_positions, &chain_structures, &chain_colors, &mut ribbon_vertices, &mut ribbon_indices);
                             chain_positions.clear();
                             chain_structures.clear();
                             chain_colors.clear();
                         }
                         if !chain_positions.is_empty() {
                             // Debug loop transitions
                             // println!("Flushing Chain '{}' ({} atoms). Next: '{}' (Res {})", current_chain_id, chain_positions.len(), chain, residue_id);
                             flush_chain(&chain_positions, &chain_structures, &chain_colors, &mut ribbon_vertices, &mut ribbon_indices);
                             chain_positions.clear();
                             chain_structures.clear();
                             chain_colors.clear();
                         }
                         current_chain_id = chain.clone();
                    }

                    last_residue_id = atom_res_id;
                    
                    chain_positions.push(self.system.positions[i]);
                    
                    let structure = if is_nucleic {
                        // Force Nucleic Acids to render as "Sheet" (Flat Ribbon) for now
                        atom_core::SecondaryStructure::Sheet 
                    } else if i < self.system.secondary_structure.len() { 
                        self.system.secondary_structure[i] 
                    } else { 
                        atom_core::SecondaryStructure::Unknown 
                    };
                    chain_structures.push(structure);
                    
                     let color = match self.color_scheme {
                        ColorScheme::Element => glam::Vec3::new(0.5, 0.5, 0.5), 
                        ColorScheme::Chain => if i < self.system.chain_ids.len() { color_by_chain(&self.system.chain_ids[i]) } else { glam::Vec3::ZERO },
                        ColorScheme::ResidueType => if i < self.system.residue_names.len() { color_by_residue(&self.system.residue_names[i]) } else { glam::Vec3::ZERO },
                         ColorScheme::SecondaryStructure => {
                              if i < self.system.secondary_structure.len() {
                                  match self.system.secondary_structure[i] {
                                      atom_core::SecondaryStructure::Helix => glam::Vec3::new(1.0, 0.0, 1.0),
                                      atom_core::SecondaryStructure::Sheet => glam::Vec3::new(1.0, 1.0, 0.0),
                                      atom_core::SecondaryStructure::Loop => glam::Vec3::new(0.9, 0.9, 0.9),
                                      atom_core::SecondaryStructure::Unknown => glam::Vec3::new(0.5, 0.5, 0.5),
                                  }
                              } else { glam::Vec3::new(0.5, 0.5, 0.5) }
                         },
                         ColorScheme::Hydrophobicity => if i < self.system.residue_names.len() { color_by_hydrophobicity(&self.system.residue_names[i]) } else { glam::Vec3::new(0.5, 0.5, 0.5) },
                     };
                    chain_colors.push(color);
                }
            }
            // Flush last
            if !chain_positions.is_empty() {
                flush_chain(&chain_positions, &chain_structures, &chain_colors, &mut ribbon_vertices, &mut ribbon_indices);
            }

            // Recreate buffers to fit data exactly
            self.ribbon_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ribbon Vertex Buffer"),
                contents: bytemuck::cast_slice(&ribbon_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.ribbon_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ribbon Index Buffer"),
                contents: bytemuck::cast_slice(&ribbon_indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
            self.num_ribbon_indices = ribbon_indices.len() as u32;

        } else {
            // Clear ribbon if not in mode (optional, but good for safety)
            self.num_ribbon_indices = 0;
        }
    }

    fn select_atom_from_ray(&mut self, ray: crate::camera::Ray) {
        let mut min_dist = f32::MAX;
        let mut closest = None;
        
        let atom_radius_sq = 1.0; // Generous hit radius (1.0 Angstrom squared)

        for (i, &pos) in self.system.positions.iter().enumerate() {
            // Ray-Sphere Intersection Test (Simplified)
            // Vector from Ray Origin to Sphere Center
            let oc = pos - ray.origin;
            let projection = oc.dot(ray.direction);
            
            // If projection is negative, sphere is behind ray
            if projection < 0.0 { continue; }
            
            let closest_point = ray.origin + ray.direction * projection;
            let dist_sq = closest_point.distance_squared(pos);
            
            if dist_sq < atom_radius_sq {
                let dist_to_camera = oc.length_squared();
                if dist_to_camera < min_dist {
                    min_dist = dist_to_camera;
                    closest = Some(i);
                }
            }
        }
        
        if let Some(atom_idx) = closest {
            println!("Clicked Atom: {} ({})", atom_idx, element_name(self.system.atomic_numbers[atom_idx]));
            
            if self.measurement_mode {
                // Measurement Logic
                if !self.measurement_click_buffer.contains(&atom_idx) {
                    self.measurement_click_buffer.push(atom_idx);
                    println!("Measurement: Point {} set.", self.measurement_click_buffer.len());

                    if self.measurement_click_buffer.len() == 2 {
                         let a = self.measurement_click_buffer[0];
                         let b = self.measurement_click_buffer[1];
                         let dist = self.system.positions[a].distance(self.system.positions[b]);
                         self.measurements.push((a, b, dist));
                         println!("📏 Distance between #{} and #{}: {:.3} Å", a, b, dist);
                         
                         self.measurement_click_buffer.clear();
                    }
                }
            } else {
                // Normal Selection Logic
                self.selected_atom = Some(atom_idx); // Store selection for UI logic
    
                // Build Adjacency and Find Ligand
                let adj = self.system.build_adjacency_list();
                let connected = self.system.find_connected_component(atom_idx, &adj);
                
                println!("Selected Molecule: {} atoms", connected.len());
                self.selected_component = connected.iter().cloned().collect(); // Store component as HashSet
    
                // Update Custom Colors to Highlight
                self.custom_colors.clear(); // Clear previous selection
                for &idx in &connected {
                    self.custom_colors.insert(idx, [0.0, 1.0, 0.0]); // Bright Green Highlight
                }
                // Highlight clicked atom specifically
                self.custom_colors.insert(atom_idx, [1.0, 0.0, 0.0]); // Red for clicked center
                
                self.needs_buffer_update = true;
            }
        } else {
             // Clicked nothing -> Deselect
             if !self.custom_colors.is_empty() || self.selected_atom.is_some() {
                 println!("Cleared Selection");
                 self.custom_colors.clear();
                 self.selected_atom = None; 
                 self.selected_component.clear();
                 self.is_isolated = false; // Exit isolation on deselect
                 self.needs_buffer_update = true;
             }
        }
    }
    


    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.1, // Darker background
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw Atoms
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
            
            // Draw Bonds (Same pipeline, different geometry/instances)
            if self.num_bonds > 0 {
                render_pass.set_vertex_buffer(0, self.cylinder_vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.bond_instance_buffer.slice(..));
                render_pass.set_index_buffer(self.cylinder_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_cylinder_indices, 0, 0..self.num_bonds);
            }

            // Draw Ribbon (Cartoon)
            if self.render_mode == RenderMode::Ribbon && self.num_ribbon_indices > 0 {
                render_pass.set_pipeline(&self.ribbon_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.ribbon_vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.ribbon_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_ribbon_indices, 0, 0..1);
            }

            // Draw Surface
            if self.render_mode == RenderMode::Surface && self.num_surface_indices > 0 {
                if let (Some(vb), Some(ib)) = (&self.surface_vertex_buffer, &self.surface_index_buffer) {
                    render_pass.set_pipeline(&self.surface_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..self.num_surface_indices, 0, 0..1);
                }
            }
        }
        
        // Egui Render Pass
        let egui_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_ctx.clone(); // Clone context to avoid borrowing self
        ctx.begin_frame(egui_input);

        ui::draw_ui(self, &ctx);


        let full_output = ctx.end_frame();
        let _convert_output = |_output: egui::PlatformOutput| {
             // Dummy conversion if needed or handled by generic winit integration
        };
        // self.egui_state.handle_platform_output(&self.window, full_output.platform_output); // Winit 0.29 logic might differ
        
        // Egui Primitives
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point); 
        
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }
        
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );
        
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            self.egui_renderer.render(&mut pass, &clipped_primitives, &screen_descriptor);
        }
        
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    pub fn save_screenshot(&mut self) {
        // Note: This is a simplified placeholder. Full implementation would require
        // capturing the render target texture, which needs async texture readback.
        // For now, we just notify the user.
        // let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        // let filename = format!("atom_screenshot_{}.png", timestamp);
        // println!("📸 Screenshot saved: {}", filename);
        // println!("(Note: Full texture capture not yet implemented - this is a placeholder)");

        // Correction: WGPU swapchain textures cannot be read back directly on all backends easily.
        // Easier: Render to an offscreen texture, blit to screen, and read offscreen.
        // OR: use `copy_texture_to_buffer` on the current frame before presenting.
        // But `render` consumes the surface texture.
        
        println!("📸 Screenshot logic to be implemented via separate render pass or offscreen buffer.");
        // For simplicity in this step, we'll implement a dedicated 'capture' function that runs a separate render pass 
        // to a mappable texture.
        self.capture_frame();
    }

    fn capture_frame(&self) {
        let width = self.config.width;
        let height = self.config.height;
        
        // 1. Create Texture
        let texture_desc = wgpu::TextureDescriptor {
             size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
             mip_level_count: 1,
             sample_count: 1,
             dimension: wgpu::TextureDimension::D2,
             format: wgpu::TextureFormat::Rgba8UnormSrgb, // Use standard format
             usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
             label: Some("Capture Texture"),
             view_formats: &[],
        };
        let texture = self.device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // 2. Render to this texture (Duplicate render pass logic - simplified)
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Capture Encoder") });
        {
             let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Capture Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view, // Use self.depth_view instead of self.depth_texture.view
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
            
            // Draw Bonds
            if self.num_bonds > 0 { // Removed !self.is_isolated as it's not in the original render logic
                 render_pass.set_vertex_buffer(0, self.cylinder_vertex_buffer.slice(..)); // Added cylinder vertex buffer
                 render_pass.set_vertex_buffer(1, self.bond_instance_buffer.slice(..));
                 render_pass.set_index_buffer(self.cylinder_index_buffer.slice(..), wgpu::IndexFormat::Uint16); // Added cylinder index buffer
                 render_pass.draw_indexed(0..self.num_cylinder_indices, 0, 0..self.num_bonds); // Changed num_indices to num_cylinder_indices
            }

            // Draw Ribbon (Cartoon)
            if self.render_mode == RenderMode::Ribbon && self.num_ribbon_indices > 0 {
                render_pass.set_pipeline(&self.ribbon_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.ribbon_vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.ribbon_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_ribbon_indices, 0, 0..1);
            }
        }
        
        // 3. Copy to Buffer
        let u32_size = std::mem::size_of::<u32>() as u32;
        let output_buffer_size = (u32_size * width * height) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: Some("Output Buffer"),
            mapped_at_creation: false,
        };
        let output_buffer = self.device.create_buffer(&output_buffer_desc);
        
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(u32_size * width),
                    rows_per_image: Some(height),
                },
            },
            texture_desc.size,
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // 4. Map and Save
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            
            let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
            let filename = format!("screenshot_{}.png", timestamp);
            
            use image::{ImageBuffer, Rgba};
            let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data.to_vec()).unwrap();
            buffer.save(&filename).unwrap();
            
            println!("📸 Saved screenshot to {}", filename);
        }
        output_buffer.unmap();
    }
}




pub async fn run(initial_system: System, force_field: Option<Box<dyn ForceField>>, custom_colors: std::collections::HashMap<usize, [f32; 3]>) {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    window.set_title("Atom Engine - Live Physics");

    let mut state = State::new(&window, initial_system, force_field, custom_colors).await;

    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    ..
                                },
                            ..
                        } => control_flow.exit(),
                    // Drag and Drop File Loading
                    WindowEvent::DroppedFile(path) => {
                        println!("File Dropped: {:?}", path);
                        match atom_core::System::load_from_file(&path) {
                            Ok(new_system) => {
                                println!("Successfully loaded {} atoms!", new_system.positions.len());
                                state.system = new_system;
                                
                                // Reset Camera
                                let center = state.system.box_size * 0.5;
                                state.camera.target = center;
                                state.camera.eye = glam::Vec3::new(center.x, center.y, center.z + 40.0);
                                
                                // Rebuild GPU Buffers
                                state.rebuild_buffers();
                                state.window.request_redraw();
                            }
                            Err(e) => {
                                eprintln!("Failed to load file: {}", e);
                            }
                        }
                    },
                    // Focus 'F'
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::KeyF),
                                ..
                            },
                        ..
                    } => {
                        let center = state.system.box_size * 0.5;
                        state.camera.target = center;
                        state.camera.eye = glam::Vec3::new(center.x, center.y, center.z + 40.0);
                        state.window.request_redraw();
                    },
                    // Grab Pivot 'G' - Set target to nearest atom to camera
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::KeyG),
                                ..
                            },
                        ..
                    } => {
                        let eye = state.camera.eye;
                        let mut closest_dist = f32::MAX;
                        let mut closest_pos = state.camera.target;
                        
                        for pos in &state.system.positions {
                            let dist = (*pos - eye).length_squared();
                            if dist < closest_dist {
                                closest_dist = dist;
                                closest_pos = *pos;
                            }
                        }
                        println!("Pivoting to nearest atom at {:?}", closest_pos);
                        state.camera.target = closest_pos;
                        state.window.request_redraw();
                    },
                    // Remove Atom 'X'
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::KeyX),
                                ..
                            },
                        ..
                    } => {
                        if !state.system.positions.is_empty() {
                            println!("Deleting Atom index {}", state.system.positions.len() - 1);
                            state.system.positions.pop();
                            if !state.system.atomic_numbers.is_empty() { state.system.atomic_numbers.pop(); }
                            if !state.system.masses.is_empty() { state.system.masses.pop(); }
                            if !state.system.forces.is_empty() { state.system.forces.pop(); }
                            if !state.system.velocities.is_empty() { state.system.velocities.pop(); }
                            
                            // Cleanup Bonds
                            let max_idx = state.system.positions.len() as u32;
                            state.system.bonds.retain(|&(i, j, _)| i < max_idx && j < max_idx);
                            
                            state.should_rebuild_buffers = true; 
                            state.needs_buffer_update = true; // Ensure one update
                        }
                    },
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                state.window().request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}

impl<'a> State<'a> {
    pub fn rebuild_buffers(&mut self) {
        use wgpu::util::DeviceExt;
        let device = &self.device; // Borrow device
        
        // 1. Rebuild Atom Buffers
        let n_atoms = self.system.positions.len();
        // Create instance buffer
        let instances = (0..n_atoms).map(|i| {
             let pos = self.system.positions[i];
             let atomic_number = if i < self.system.atomic_numbers.len() { self.system.atomic_numbers[i] } else { 6 };
            
            // CPK Coloring
            let color = if let Some(c) = self.custom_colors.get(&i) {
                glam::Vec3::from(*c)
            } else {
                match atomic_number {
                    1 => glam::Vec3::new(1.0, 1.0, 1.0), // H
                    6 => glam::Vec3::new(0.2, 0.2, 0.2), // C
                    7 => glam::Vec3::new(0.0, 0.0, 1.0), // N
                    8 => glam::Vec3::new(1.0, 0.0, 0.0), // O
                    16 => glam::Vec3::new(1.0, 1.0, 0.0), // S
                    15 => glam::Vec3::new(1.0, 0.65, 0.0), // P
                    _ => glam::Vec3::new(0.8, 0.0, 0.8), // Magenta
                }
            };
            
             // Scale based on Atomic Number (VDW Radius approx) + Global Scale for visibility
            let radius = match atomic_number {
                 1 => 1.2,
                 6 => 1.7,
                 7 => 1.55,
                 8 => 1.52,
                 15 => 1.8,
                 16 => 1.8,
                 _ => 1.7,
            };
            // "Ball-and-Stick" style scale
            let scale = radius * 0.4;

            Instance { position: pos, color, scale }
        }).collect::<Vec<_>>();
        
        // Convert to Raw for GPU
        let instance_raw = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_data = bytemuck::cast_slice(&instance_raw);
        
        self.instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: instance_data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.num_instances = n_atoms as u32;

        // 2. Rebuild Bond Buffers
        // Just create empty max size buffer for now
        let max_bonds = 200_000;
         self.bond_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bond Instance Buffer"),
            size: (max_bonds * std::mem::size_of::<InstanceRaw>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 3. Rebuild Ribbon Buffers (Dynamic Size)
    // Estimate: 64 verts per atom (conservative) and 256 indices per atom to handle high-res splines/biological assemblies
    let estimated_verts = (n_atoms as u64 * 64).max(2_000_000);
    let estimated_inds = (n_atoms as u64 * 256).max(6_000_000);
    
    self.ribbon_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ribbon Vertex Buffer"),
        size: estimated_verts * std::mem::size_of::<mesh::RibbonVertex>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    self.ribbon_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
         label: Some("Ribbon Index Buffer"),
         size: estimated_inds * std::mem::size_of::<u32>() as wgpu::BufferAddress, // Use u32 for indices
         usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
         mapped_at_creation: false,
    });
        
        println!("Buffers Rebuilt for {} atoms.", n_atoms);
    }
}
