use glam::{Vec3, Vec2};
use crate::spline::CatmullRomSpline;
use atom_core::SecondaryStructure;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RibbonVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl RibbonVertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RibbonVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, offset: 0, format: wgpu::VertexFormat::Float32x3 }, // position
                wgpu::VertexAttribute { shader_location: 1, offset: 12, format: wgpu::VertexFormat::Float32x3 }, // normal
                wgpu::VertexAttribute { shader_location: 2, offset: 24, format: wgpu::VertexFormat::Float32x3 }, // color
            ],
        }
    }
}

pub struct RibbonMeshGenerator;

impl RibbonMeshGenerator {
    pub fn generate_chain_mesh(
        positions: &[Vec3],
        structures: &[SecondaryStructure],
        colors: &[Vec3],
        subdivisions: usize
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        if positions.len() < 2 {
            return (Vec::new(), Vec::new());
        }

        let spline = CatmullRomSpline::new(positions.to_vec());
        let mut vertices = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Parameters
        let tube_radius = 0.25;
        let ribbon_width = 1.2;
        let ribbon_thickness = 0.2;
        let segments = positions.len() * subdivisions;

        // Shape Profiles (2D offsets)
        // Tube: Circle
        let circle_res = 8;
        let circle_shape: Vec<Vec2> = (0..circle_res).map(|i| {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (circle_res as f32);
            Vec2::new(angle.cos() * tube_radius, angle.sin() * tube_radius)
        }).collect();

        // Ribbon: Flat Rectangle
        let ribbon_shape = vec![
            Vec2::new(ribbon_width * 0.5, ribbon_thickness * 0.5),
            Vec2::new(-ribbon_width * 0.5, ribbon_thickness * 0.5),
            Vec2::new(-ribbon_width * 0.5, -ribbon_thickness * 0.5),
            Vec2::new(ribbon_width * 0.5, -ribbon_thickness * 0.5),
        ];

        // Frame Tracking
        let mut prev_normal = Vec3::X; 

        for s in 0..segments {
            let t = s as f32 / subdivisions as f32;
            let idx = t.floor() as usize;
            
            // Determine structure type (Helix/Sheet/Loop) at this point
            let structure = if idx < structures.len() { structures[idx] } else { SecondaryStructure::Loop };
            let color = if idx < colors.len() { colors[idx] } else { Vec3::ONE };

            let pos = spline.interpolate(t);
            let tangent = spline.tangent(t);
            
            // Calculate Frame (Normal/Binormal)
            // Ideally use Parallel Transport or Reference to neighbors
            // Simple up vector strategy:
            let up_guess = if tangent.cross(Vec3::Y).length_squared() > 0.01 { Vec3::Y } else { Vec3::Z };
            let right = tangent.cross(up_guess).normalize();
            let up = right.cross(tangent).normalize();
            
            // For ribbons, aligning 'up' to the curvature usually looks best for helices
            // But for simple robust rendering, we'll stick to a minimizing twist approach or fixed up
            if s > 0 {
                // transport prev_normal
                 let _axis = prev_normal.cross(up); // Rotation axis to align? 
                 // Simple hack: Just use the calculated one for now, smoothing comes later
            }
            prev_normal = up;

            // Choose Profile
            let profile = match structure {
                SecondaryStructure::Helix | SecondaryStructure::Sheet => &ribbon_shape,
                _ => &circle_shape,
            };

            // Extrude Profile
            let start_v_idx = vertices.len() as u16;
            for point2d in profile {
                // Transform 2D point to 3D frame
                // Profile X -> Right, Profile Y -> Up
                let offset = right * point2d.x + up * point2d.y;
                let v_pos = pos + offset;
                
                vertices.push(RibbonVertex {
                    position: v_pos.to_array(),
                    normal: offset.normalize_or_zero().to_array(), // Approx normal
                    color: color.to_array(),
                });
            }

            // Generate Indices (Connect to previous ring)
            if s > 0 {
                let ring_size = profile.len() as u16;
                let _prev_ring_start = start_v_idx - ring_size;
                
                // Assuming topology is consistent (handling transitions is tricky, for now assuming constant profile size per segment or separate draw calls)
                // TODO: Transitions between shapes (tube -> ribbon) require matching vertex counts.
                // SIMPLIFICATION: If shape changes, we break the mesh or degenerate triangles.
                // For MVP: We will treat everything as a Tube but Scale it flat for ribbons.
                
                // Re-implementation with Consistent Topology (8-sided tube for everything)
                // Just squash it for ribbons.
            }
        }
        
        // RE-DOING LOOP FOR CONSISTENT TOPOLOGY
        vertices.clear();
        indices.clear();
        
        // Consistent 8-sided tube for everything, squashed for ribbons
        let sides = 8;
        
        for s in 0..segments {
             let t = s as f32 / subdivisions as f32;
             let idx = t.floor() as usize;
             let structure = if idx < structures.len() { structures[idx] } else { SecondaryStructure::Loop };
             let color = if idx < colors.len() { colors[idx] } else { Vec3::ONE };
 
             let pos = spline.interpolate(t);
             let tangent = spline.tangent(t);
             
             // Robust Frame (Parallel Transport approximation)
             let up_guess = if tangent.cross(Vec3::Y).length_squared() > 0.01 { Vec3::Y } else { Vec3::Z };
             let right = tangent.cross(up_guess).normalize();
             let up = right.cross(tangent).normalize();
             
             // Scale factors
             let (scale_x, scale_y) = match structure {
                 SecondaryStructure::Helix => (1.2, 0.2), // Wide/Thin
                 SecondaryStructure::Sheet => (1.2, 0.2), // Wide/Thin
                 _ => (0.3, 0.3), // Round Tube
             };
             
             let start_v = vertices.len() as u32;
             
             for i in 0..sides {
                 let angle = 2.0 * std::f32::consts::PI * (i as f32) / (sides as f32);
                 let x = angle.cos();
                 let y = angle.sin();
                 
                 let local_pos = right * (x * scale_x) + up * (y * scale_y);
                 let normal = (right * x + up * y).normalize(); // Approximation
                 
                 vertices.push(RibbonVertex {
                     position: (pos + local_pos).to_array(),
                     normal: normal.to_array(),
                     color: color.to_array(),
                 });
             }
             
             if s > 0 {
                 let prev_start = start_v - sides as u32;
                 for i in 0..sides {
                     let next = (i + 1) % sides;
                     let curr = i;
                     
                     // Quad: prev_curr, prev_next, curr_next, curr_curr
                     let pc = prev_start + curr as u32;
                     let pn = prev_start + next as u32;
                     let cc = start_v + curr as u32;
                     let cn = start_v + next as u32;
                     
                     indices.push(pc); indices.push(cc); indices.push(pn);
                     indices.push(pn); indices.push(cc); indices.push(cn);
                 }
             }
        }

        (vertices, indices)
    }
}
