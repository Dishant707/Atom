// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>, // Added view position for specular/rim calculation
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    
    var out: VertexOutput;
    out.color = instance.color;
    
    let world_pos = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_pos = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    
    // Transform normal (assuming uniform scale, otherwise check inverse transpose)
    // Taking 3x3 rotation from model matrix
    let rot_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    out.world_normal = normalize(rot_matrix * model.normal);
    
    return out;
}

struct RibbonVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_ribbon(
    model: RibbonVertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color; // Pass color, light in fragment
    out.world_normal = model.normal;
    out.world_pos = model.position; // Approximate
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Lighting Parameters
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.8)); // Sun position
    let view_dir = normalize(camera.view_pos.xyz - in.world_pos);
    let normal = normalize(in.world_normal);

    // 1. Ambient
    let ambient_strength = 0.25;
    let ambient = ambient_strength * in.color;
    
    // 2. Diffuse (Lambert)
    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = diff * in.color * 0.9;
    
    // 3. Specular (Blinn-Phong)
    let specular_strength = 0.3;
    let halfway_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0);
    let specular = vec3<f32>(1.0) * spec * specular_strength;
    
    // 4. Rim Light (Fresnel-like)
    // Adds "glow" or outline effect at grazing angles
    let rim_strength = 0.45;
    let rim_amount = 1.0 - max(dot(view_dir, normal), 0.0);
    let rim = vec3<f32>(1.0) * pow(rim_amount, 3.0) * rim_strength;

    // Combine
    let result = ambient + diffuse + specular + rim;
    
    // Gamma correction? (Optional, maybe for later polish)
    
    return vec4<f32>(result, 1.0);
}
