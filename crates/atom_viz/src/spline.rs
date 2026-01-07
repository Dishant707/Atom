use glam::Vec3;

pub struct CatmullRomSpline {
    points: Vec<Vec3>,
}

impl CatmullRomSpline {
    pub fn new(points: Vec<Vec3>) -> Self {
        Self { points }
    }

    /// Evaluates the spline at parameter t.
    /// t ranges from 0.0 to (points.len() - 1) as f32.
    pub fn interpolate(&self, t: f32) -> Vec3 {
        if self.points.is_empty() {
            return Vec3::ZERO;
        }
        if self.points.len() == 1 {
            return self.points[0];
        }

        let len = self.points.len() as f32;
        let p = t.clamp(0.0, len - 1.0);
        let i = p.floor() as usize;
        let local_t = p - i as f32;

        let p0 = if i == 0 { self.points[0] } else { self.points[i - 1] };
        let p1 = self.points[i];
        let p2 = if i + 1 >= self.points.len() { self.points[i] } else { self.points[i + 1] };
        let p3 = if i + 2 >= self.points.len() { p2 } else { self.points[i + 2] };

        Self::catmull_rom(p0, p1, p2, p3, local_t)
    }
    
    /// Evaluates the tangent (derivative) at parameter t.
    pub fn tangent(&self, t: f32) -> Vec3 {
        if self.points.is_empty() { return Vec3::Y; }
        if self.points.len() == 1 { return Vec3::Y; }
        
        let len = self.points.len() as f32;
        let p = t.clamp(0.0, len - 1.0);
        let i = p.floor() as usize;
        let local_t = p - i as f32;

        let p0 = if i == 0 { self.points[0] } else { self.points[i - 1] };
        let p1 = self.points[i];
        let p2 = if i + 1 >= self.points.len() { self.points[i] } else { self.points[i + 1] };
        let p3 = if i + 2 >= self.points.len() { p2 } else { self.points[i + 2] };

        Self::catmull_rom_derivative(p0, p1, p2, p3, local_t).normalize_or_zero()
    }

    fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
        let t2 = t * t;
        let t3 = t2 * t;

        let v0 = (p2 - p0) * 0.5;
        let v1 = (p3 - p1) * 0.5;

        (2.0 * p1 - 2.0 * p2 + v0 + v1) * t3
            + (-3.0 * p1 + 3.0 * p2 - 2.0 * v0 - v1) * t2
            + v0 * t
            + p1
    }
    
    fn catmull_rom_derivative(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
        let t2 = t * t;

        let v0 = (p2 - p0) * 0.5;
        let v1 = (p3 - p1) * 0.5;

        (2.0 * p1 - 2.0 * p2 + v0 + v1) * 3.0 * t2
            + (-3.0 * p1 + 3.0 * p2 - 2.0 * v0 - v1) * 2.0 * t
            + v0
    }
}
