// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) view_dir: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = model.position;
    out.world_normal = model.normal;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.view_dir = normalize(camera.view_pos.xyz - model.position);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(in.view_dir);
    let L = normalize(vec3<f32>(1.0, 1.0, 1.0)); // Light direction (Directional)

    // Base Color (White/Light Grey default, can map values later)
    let base_color = vec3<f32>(0.9, 0.9, 0.95);

    // Blinn-Phong Shading
    let ambient = 0.2 * base_color;
    
    let diff = max(dot(N, L), 0.0);
    let diffuse = diff * base_color;
    
    let H = normalize(L + V);
    let spec = pow(max(dot(N, H), 0.0), 32.0);
    let specular = vec3<f32>(0.3) * spec; // Reduced spec intensity
    
    // Fresnel Effect (Rim Lighting) for "Glassy" feel
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    let rim = vec3<f32>(0.2, 0.3, 0.4) * fresnel; // Blue-ish rim

    let result = ambient + diffuse + specular + rim;

    return vec4<f32>(result, 1.0); // Opaque for now, can perform transparency
}
