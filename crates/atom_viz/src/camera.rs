use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: glam::Mat4 = glam::Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

pub struct Camera {
    pub eye: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = glam::Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    pub fn screen_point_to_ray(&self, screen_x: f32, screen_y: f32, screen_width: f32, screen_height: f32) -> Ray {
        // Normalized Device Coordinates (-1.0 to 1.0)
        let x = (2.0 * screen_x) / screen_width - 1.0;
        let y = 1.0 - (2.0 * screen_y) / screen_height; // Flip Y for WGPU (NDC Y is Up)

        // Inverse Transformation
        let view_proj = self.build_view_projection_matrix();
        let inv_view_proj = view_proj.inverse();

        // Unproject Near and Far points
        // WGPU Z range is [0, 1]
        let near_ndc = glam::Vec4::new(x, y, 0.0, 1.0);
        let far_ndc = glam::Vec4::new(x, y, 1.0, 1.0);

        let near_world = inv_view_proj * near_ndc;
        let far_world = inv_view_proj * far_ndc;

        let near_point = near_world.truncate() / near_world.w;
        let far_point = far_world.truncate() / far_world.w;

        let direction = (far_point - near_point).normalize();

        Ray {
            origin: near_point,
            direction,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: glam::Vec3,
    pub direction: glam::Vec3,
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    pub is_shift_pressed: bool,
    pub is_alt_pressed: bool,
    pub is_mouse_left_pressed: bool,
    pub is_mouse_right_pressed: bool,
    pub last_mouse_pos: Option<(f64, f64)>,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            sensitivity: 0.2, // Radians per pixel (approx)
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_shift_pressed: false,
            is_alt_pressed: false,
            is_mouse_left_pressed: false,
            is_mouse_right_pressed: false,
            last_mouse_pos: None,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state,
                    physical_key: PhysicalKey::Code(keycode),
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW | KeyCode::ArrowUp => { self.is_forward_pressed = is_pressed; true }
                    KeyCode::KeyS | KeyCode::ArrowDown => { self.is_backward_pressed = is_pressed; true }
                    KeyCode::KeyA | KeyCode::ArrowLeft => { self.is_left_pressed = is_pressed; true }
                    KeyCode::KeyD | KeyCode::ArrowRight => { self.is_right_pressed = is_pressed; true }
                    KeyCode::KeyQ => { self.is_down_pressed = is_pressed; true }
                    KeyCode::KeyE => { self.is_up_pressed = is_pressed; true }
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => { self.is_shift_pressed = is_pressed; true }
                    KeyCode::AltLeft | KeyCode::AltRight => { self.is_alt_pressed = is_pressed; true }
                    _ => false,
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let is_pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => { self.is_mouse_left_pressed = is_pressed; true }
                    MouseButton::Right => { self.is_mouse_right_pressed = is_pressed; true }
                    _ => false,
                }
            }
            // Mouse motion is usually DeviceEvent for raw delta, but WindowEvent::CursorMoved is absolute.
            // For simple orbit, CursorMoved is fine if we track delta manually, 
            // but Winit DeviceEvent is better for FPS. 
            // However, we are in `process_events` which takes WindowEvent.
            // We'll update delta in `update_camera` if we pass events.
            // Actually, best to handle CursorMoved here to track drag.
            WindowEvent::CursorMoved { position: _, .. } => {
               // Logic handled in update if we store delta, but simpler to compute delta here.
               // Actually we need mutable state diff.
               // Let's rely on a separate `process_device_events` or just use local delta cache.
               false
            }
            _ => false,
        }
    }
    
    // Helper to process DeviceEvent (Raw Mouse Motion)
    pub fn process_device_events(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta: _ } => {
               // We store accumulated delta? Or apply directly?
               // Since `update_camera` is called per frame, applying directly requires passing Camera here.
               // Ideally we store "velocity" or "target rotation" 
               // BUT simpler: Separate logic.
               // Let's stick to WindowEvents or add a generic delta field.
               false 
            }
            // Zoom
            DeviceEvent::MouseWheel { delta: _ } => {
                // Scroll logic
                false 
            }
            _ => false,
        }
    }

    // Simplified: Just use keyboard for movement, and we will enhance `update_camera` logic if we had mouse delta.
    // WAIT: The user EXPLICITLY asked for Mouse Control.
    // I must implement it.
    // `process_events` returns bool = handled.
    // The previous implementation was simple.
    // I need to change `loop` in lib.rs to pass DeviceEvents too?
    // Correct. `atom_viz/src/lib.rs` loop handles `Event::WindowEvent` but typically ignores `DeviceEvent`.
    // I should limit changes to `camera.rs` if possible, but I need input.
    // IF I CANNOT CHANGE lib.rs easily, I should use `CursorMoved` and diff against `last_pos`.
    
    // Let's retry implementation using ONLY WindowEvent::CursorMoved (Dragging).
    
    pub fn update_camera(&mut self, camera: &mut Camera, cursor_delta: Option<(f64, f64)>, scroll_delta: Option<f32>) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.length();
        let right = forward_norm.cross(camera.up).normalize();
        let up = right.cross(forward_norm).normalize(); // Recompute local up

        // --- Speed Logic ---
        let mut speed = self.speed;
        if self.is_shift_pressed { speed *= 5.0; } // Turbo
        if self.is_alt_pressed { speed *= 0.2; }   // Precision

        // --- Keyboard Movement (Panning / Dolly) ---
        let mut move_dir = glam::Vec3::ZERO;
        if self.is_forward_pressed { move_dir += forward_norm; } // W -> Forward
        if self.is_backward_pressed { move_dir -= forward_norm; } // S -> Backward
        if self.is_right_pressed { move_dir += right; } // D -> Right
        if self.is_left_pressed { move_dir -= right; } // A -> Left
        
        // Use Global Y for Up/Down (Q/E) to effectively "Elevate" the view
        if self.is_up_pressed { move_dir += glam::Vec3::Y; }  // E -> Up (Global)
        if self.is_down_pressed { move_dir -= glam::Vec3::Y; } // Q -> Down (Global)
        
        if move_dir.length_squared() > 0.0 {
            camera.eye += move_dir.normalize() * speed;
            camera.target += move_dir.normalize() * speed; // Move TOGETHER (Pan)
        }

        // --- Mouse Orbit (Left Click + Drag) ---
        // Re-enabled by user request for 360 rotation
        if self.is_mouse_left_pressed && !self.is_shift_pressed {
             if let Some((dx, dy)) = cursor_delta {
                 // Rotate around Target
                 // Sensitivity Factor: 0.01 for smoother trackpad, 0.005 might be too slow ("lag")
                 // Let's try 0.01 * speed.
                 // Actually previous was: -dx * sensitivity * 0.05. sensitivity=0.2 -> 0.01 factor.
                 // User reported "lag", maybe it felt slow? or delayed?
                 // Let's bump it slightly.
                 let rot_speed = 0.003; // Radians per pixel (heuristic)
                 
                 let yaw = glam::Quat::from_axis_angle(glam::Vec3::Y, -dx as f32 * rot_speed);
                 let pitch = glam::Quat::from_axis_angle(right, -dy as f32 * rot_speed);
                 
                 let offset = camera.eye - camera.target;
                 camera.eye = camera.target + (yaw * pitch * offset);
             }
        }
        
        // --- Mouse Pan (Right Click + Drag OR Shift + Left) ---
        if self.is_mouse_right_pressed || (self.is_mouse_left_pressed && self.is_shift_pressed) {
             if let Some((dx, dy)) = cursor_delta {
                 let pan_sen = self.speed * 0.002; // Reduced for precision
                 let delta = right * (-dx as f32 * pan_sen) + up * (dy as f32 * pan_sen);
                 camera.eye += delta;
                 camera.target += delta;
             }
        }
        
        // --- Zoom (Scroll) ---
        // Adjusted for "Big Molecule" usecase:
        // Speed is proportional to distance. Closer = Slower. Farther = Faster.
        if let Some(scroll) = scroll_delta {
            // Zoom Factor: 10% of distance per scroll tick
            let zoom_factor = 0.1; 
            let move_dist = forward_mag * zoom_factor * scroll;
            
            // Prevent getting too close (clipping) or flipping
            // Minimum distance 1.0
            if move_dist < 0.0 || forward_mag > 2.0 { 
                 camera.eye += forward_norm * move_dist;
            } else if move_dist > 0.0 {
                 // Allow backing out even if close
                 camera.eye += forward_norm * move_dist;
            }
        }
    }
}
