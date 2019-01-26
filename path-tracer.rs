#!/usr/bin/env run-cargo-script
//!```cargo
//![package]
//!edition = "2018"
//!
//![dependencies]
//!image = "*"
//!rayon = "*"
//!rand = "*"
//!```
///
/// This is simple path tracer implementation, with following features
/// - Only lambertian diffusion for now
/// - Signed distance function defined objects
/// - Ray marching tracing
///
use image::{ImageBuffer, Rgb, RgbImage};
use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

// -----------------------------------------------------------------------------
// Vector
// -----------------------------------------------------------------------------
type Scalar = f32;
const EPSILON: Scalar = 0.0001;
const MAX_MARCHING_STEPS: usize = 1024;
const MIN_DISTANCE: Scalar = 10.0 * EPSILON;
const MAX_DISTANCE: Scalar = 10000.0;
const PI: Scalar = std::f32::consts::PI;

/// Three dimentional vector
#[derive(Copy, Clone, Debug)]
pub struct Vec3([Scalar; 3]);

impl Vec3 {
    pub const ZERO: Self = Vec3([0.0; 3]);
    pub const ONE: Self = Vec3([1.0; 3]);

    #[inline]
    pub fn len(&self) -> Scalar {
        self.dot(*self).sqrt()
    }

    #[inline]
    pub fn abs(&self) -> Self {
        let [x, y, z] = self.0;
        Vec3([x.abs(), y.abs(), z.abs()])
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        *self / self.len()
    }

    #[inline]
    pub fn reflect(&self, normal: Vec3) -> Self {
        normal * self.dot(normal) * 2.0 - *self
    }

    /// Inner product
    #[inline]
    pub fn dot(&self, other: Vec3) -> Scalar {
        let v0 = self.0;
        let v1 = other.0;
        v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]
    }

    /// Cross product
    #[inline]
    pub fn cross(&self, other: Vec3) -> Self {
        //        x   y   z
        let Vec3([a1, a2, a3]) = self;
        let Vec3([b1, b2, b3]) = other;
        Vec3([a2 * b3 - a3 * b2, -(a1 * b3 - a3 * b1), a1 * b2 - a2 * b1])
    }
}

impl Sum for Vec3 {
    #[inline]
    fn sum<I: Iterator<Item = Vec3>>(iter: I) -> Self {
        iter.fold(Vec3::ZERO, |acc, v| acc + v)
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let v0 = self.0;
        let v1 = other.0;
        Vec3([v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2]])
    }
}

impl Sub for Vec3 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        let v0 = self.0;
        let v1 = other.0;
        Vec3([v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]])
    }
}

impl Mul for Vec3 {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        let v0 = self.0;
        let v1 = other.0;
        Vec3([v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2]])
    }
}

impl Mul<Scalar> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, scalar: Scalar) -> Self::Output {
        Vec3([self.0[0] * scalar, self.0[1] * scalar, self.0[2] * scalar])
    }
}

impl Div for Vec3 {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        let Vec3([x0, y0, z0]) = self;
        let Vec3([x1, y1, z1]) = other;
        Vec3([x0 / x1, y0 / y1, z0 / z1])
    }
}

impl Div<Scalar> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn div(self, scalar: Scalar) -> Self::Output {
        Vec3([self.0[0] / scalar, self.0[1] / scalar, self.0[2] / scalar])
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    #[inline]
    fn neg(self) -> Vec3 {
        let Vec3([x, y, z]) = self;
        Vec3([-x, -y, -z])
    }
}

impl From<Vec3> for Rgb<u8> {
    fn from(color: Vec3) -> Self {
        let Vec3([x, y, z]) = color;
        Rgb([
            (x.min(1.0).max(0.0) * 255.0) as u8,
            (y.min(1.0).max(0.0) * 255.0) as u8,
            (z.min(1.0).max(0.0) * 255.0) as u8,
        ])
    }
}

#[derive(Clone, Debug)]
pub struct Mat3([Scalar; 9]);

impl Mat3 {
    pub const IDENTITY: Self = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    pub fn rot_x(angle: Scalar) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Mat3([1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c])
    }

    pub fn rot_y(angle: Scalar) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Mat3([c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c])
    }

    pub fn rot_z(angle: Scalar) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Mat3([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
    }

    pub fn transpose(&self) -> Mat3 {
        let mut mo = Mat3([0.0; 9]);
        for c in 0..3 {
            for r in 0..3 {
                mo.0[r + c * 3] = self.0[c + r * 3];
            }
        }
        mo
    }
}

impl<'a> Mul for &'a Mat3 {
    type Output = Mat3;

    #[inline]
    fn mul(self, other: &Mat3) -> Mat3 {
        let m0 = self.0;
        let m1 = other.0;
        let mut mo = Mat3([0.0; 9]);
        for l in 0..3 {
            for r in 0..3 {
                for m in 0..3 {
                    mo.0[l + 3 * r] += m0[l + 3 * m] * m1[m + r * 3];
                }
            }
        }
        mo
    }
}

impl<'a> Mul<Vec3> for &'a Mat3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, other: Vec3) -> Vec3 {
        let mut vo = [0.0; 3];
        for r in 0..3 {
            for c in 0..3 {
                vo[r] += self.0[c + r * 3] * other.0[c];
            }
        }
        Vec3([vo[0], vo[1], vo[2]])
    }
}

// -----------------------------------------------------------------------------
// Shape
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Material {
    reflectance: Vec3,
    emittance: Vec3,
}

impl Material {
    fn new(reflectance: Vec3) -> Self {
        Self {
            reflectance,
            emittance: Vec3::ZERO,
        }
    }

    fn with_emittance(&self, emittance: Vec3) -> Self {
        Self {
            emittance,
            ..self.clone()
        }
    }
}

/// Distance to nearest object in a scene
pub struct Distance {
    material: Material,
    dist: Scalar,
}

pub struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }
}

/// Hit point of casted ray
#[derive(Debug)]
pub struct Hit {
    point: Vec3,
    material: Material,
}

/// Trait representing shape
pub trait Shape {
    /// signed distance function
    fn sdf(&self, point: Vec3) -> Distance;

    /// union shape
    fn union<S: Shape>(self, other: S) -> UnionShape<Self, S>
    where
        Self: Sized,
    {
        UnionShape {
            first: self,
            second: other,
        }
    }

    /// insection shape
    fn intersect<S: Shape>(self, other: S) -> IntersectShape<Self, S>
    where
        Self: Sized,
    {
        IntersectShape {
            first: self,
            second: other,
        }
    }

    /// difference shape
    fn diff<S: Shape>(self, other: S) -> DiffShape<Self, S>
    where
        Self: Sized,
    {
        DiffShape {
            first: self,
            second: other,
        }
    }

    fn translate(self, displacement: Vec3) -> TranslateShape<Self>
    where
        Self: Sized,
    {
        TranslateShape {
            shape: self,
            displacement,
        }
    }

    /// Rotate shape with provided matrix, assumes matrix to be orthogonal
    fn rot(self, mat: Mat3) -> RotateShape<Self>
    where
        Self: Sized,
    {
        RotateShape {
            shape: self,
            mat: mat.transpose(), // same as inverse, matrix is orthogonal
        }
    }

    /// Invert shape
    fn invert(self) -> InvertShape<Self>
    where
        Self: Sized,
    {
        InvertShape { shape: self }
    }

    fn scale(self, ratio: Scalar) -> ScaleShape<Self>
    where
        Self: Sized,
    {
        ScaleShape { ratio, shape: self }
    }

    /// Cast ray to the scene
    fn cast(&self, ray: Ray, max_distance: Option<Scalar>) -> Option<Hit> {
        let mut depth: Scalar = MIN_DISTANCE;
        let max_distance = max_distance.unwrap_or(MAX_DISTANCE);
        for _ in 0..MAX_MARCHING_STEPS {
            let point = ray.origin + ray.direction * depth;
            let distance = self.sdf(point);
            if distance.dist < EPSILON * depth {
                return Some(Hit {
                    point,
                    material: distance.material,
                });
            }
            depth += distance.dist;
            if depth >= max_distance {
                return None;
            }
        }
        None
    }

    /// Estimated normal at the point
    // fn normal(&self, point: Vec3) -> Vec3 {
    //     let Vec3([x, y, z]) = point;
    //     let dx =
    //         self.sdf(Vec3([x + EPSILON, y, z])).dist - self.sdf(Vec3([x - EPSILON, y, z])).dist;
    //     let dy =
    //         self.sdf(Vec3([x, y + EPSILON, z])).dist - self.sdf(Vec3([x, y - EPSILON, z])).dist;
    //     let dz =
    //         self.sdf(Vec3([x, y, z + EPSILON])).dist - self.sdf(Vec3([x, y, z - EPSILON])).dist;
    //     Vec3([dx, dy, dz]).normalize()
    // }

    /// Calculate normal using tetrahedron technique
    /// [Normals for an SDF](http://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
    fn normal(&self, point: Vec3) -> Vec3 {
        TETRAHEDRON
            .iter()
            .fold(Vec3::ZERO, |acc, k| {
                acc + *k * self.sdf(point + *k * EPSILON).dist
            })
            .normalize()
    }
}

const TETRAHEDRON: [Vec3; 4] = [
    Vec3([1.0, -1.0, -1.0]),
    Vec3([-1.0, -1.0, 1.0]),
    Vec3([-1.0, 1.0, -1.0]),
    Vec3([1.0, 1.0, 1.0]),
];

pub struct UnionShape<S0, S1> {
    first: S0,
    second: S1,
}

impl<S0, S1> Shape for UnionShape<S0, S1>
where
    S0: Shape,
    S1: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let n0 = self.first.sdf(point);
        let n1 = self.second.sdf(point);
        // min(s0, s1)
        if n0.dist < n1.dist {
            n0
        } else {
            n1
        }
    }
}

pub struct IntersectShape<S0, S1> {
    first: S0,
    second: S1,
}

impl<S0, S1> Shape for IntersectShape<S0, S1>
where
    S0: Shape,
    S1: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let n0 = self.first.sdf(point);
        let n1 = self.second.sdf(point);
        // max(s0, s1)
        if n0.dist < n1.dist {
            n1
        } else {
            n0
        }
    }
}

pub struct DiffShape<S0, S1> {
    first: S0,
    second: S1,
}

impl<S0, S1> Shape for DiffShape<S0, S1>
where
    S0: Shape,
    S1: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let n0 = self.first.sdf(point);
        let n1 = self.second.sdf(point);
        // max(s0, -s1)
        if n0.dist < -n1.dist {
            Distance {
                dist: -n1.dist,
                ..n1
            }
        } else {
            n0
        }
    }
}

pub struct TranslateShape<S> {
    shape: S,
    displacement: Vec3,
}

impl<S> Shape for TranslateShape<S>
where
    S: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        self.shape.sdf(point - self.displacement)
    }
}

pub struct RotateShape<S> {
    shape: S,
    mat: Mat3,
}

impl<S> Shape for RotateShape<S>
where
    S: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        self.shape.sdf(&self.mat * point)
    }
}

pub struct InvertShape<S> {
    shape: S,
}

impl<S> Shape for InvertShape<S>
where
    S: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let Distance { dist, material } = self.shape.sdf(point);
        Distance {
            dist: -dist,
            material,
        }
    }
}

pub struct ScaleShape<S> {
    ratio: Scalar,
    shape: S,
}

impl<S> Shape for ScaleShape<S>
where
    S: Shape,
{
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let Distance { dist, material } = self.shape.sdf(point / self.ratio);
        Distance {
            dist: dist * self.ratio,
            material,
        }
    }
}

// -----------------------------------------------------------------------------
// Shape Implementations
// -----------------------------------------------------------------------------
pub struct Sphere {
    material: Material,
    center: Vec3,
    radius: Scalar,
}

impl Sphere {
    pub fn new(material: Material, center: Vec3, radius: Scalar) -> Self {
        Self {
            material,
            center,
            radius,
        }
    }
}

impl Shape for Sphere {
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        Distance {
            dist: (self.center - point).len() - self.radius,
            material: self.material.clone(),
        }
    }
}

struct Box {
    material: Material,
    size: Vec3,
}

impl Box {
    fn new(material: Material, size: Vec3) -> Self {
        Self { material, size }
    }
}

impl Shape for Box {
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let Vec3([x, y, z]) = point.abs() - self.size / 2.0;
        let inside_dist = x.max(y).max(z).min(0.0);
        let outsize_dist = Vec3([x.max(0.0), y.max(0.0), z.max(0.0)]).len();
        let dist = inside_dist + outsize_dist;
        Distance {
            dist,
            material: self.material.clone(),
        }
    }
}

struct Torus {
    material: Material,
    r0: Scalar,
    r1: Scalar,
}

impl Torus {
    fn new(material: Material, r0: Scalar, r1: Scalar) -> Self {
        Self { material, r0, r1 }
    }
}

impl Shape for Torus {
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let Vec3([x, y, z]) = point;
        let r0 = self.r0;
        let r1 = self.r1;
        let dist = (((x * x + z * z).sqrt() - r0).powi(2) + y * y).sqrt() - r1;
        Distance {
            dist,
            material: self.material.clone(),
        }
    }
}

struct Cylinder {
    material: Material,
    height: Scalar,
    radius: Scalar,
}

impl Cylinder {
    fn new(material: Material, height: Scalar, radius: Scalar) -> Self {
        Self {
            material,
            height,
            radius,
        }
    }
}

impl Shape for Cylinder {
    #[inline]
    fn sdf(&self, point: Vec3) -> Distance {
        let Vec3([x, y, z]) = point;
        let in_radius = (x * x + y * y).sqrt() - self.radius;
        let in_height = z.abs() - self.height / 2.0;

        let inside_dist = in_radius.max(in_height).min(0.0);
        let in_radius_neg = in_radius.max(0.0);
        let in_height_neg = in_height.max(0.0);
        let outside_dist = (in_radius_neg * in_radius_neg + in_height_neg * in_height_neg).sqrt();
        let dist = inside_dist + outside_dist;
        Distance {
            dist,
            material: self.material.clone(),
        }
    }
}

// -----------------------------------------------------------------------------
// Lights
// -----------------------------------------------------------------------------
#[derive(Debug, Copy, Clone)]
struct LightAt {
    direction: Vec3,
    intensity: Vec3,
    distance: Scalar,
}

trait Light {
    fn illuminate(&self, point: Vec3) -> LightAt;
}

struct PointLight {
    color: Vec3,
    intensity: Scalar,
    position: Vec3,
}

impl Light for PointLight {
    fn illuminate(&self, point: Vec3) -> LightAt {
        let light_dir = point - self.position;
        let r2 = light_dir.dot(light_dir);
        let distance = r2.sqrt();
        LightAt {
            direction: light_dir / distance,
            intensity: self.color * self.intensity / (4.0 * PI * r2),
            distance,
        }
    }
}

// -----------------------------------------------------------------------------
// Scene
// -----------------------------------------------------------------------------
struct Scene {
    lights: Vec<std::boxed::Box<dyn Light + Send + Sync>>,
    shape: std::boxed::Box<dyn Shape + Send + Sync>,
}

// -----------------------------------------------------------------------------
// Tracing
// -----------------------------------------------------------------------------
trait Tracer {
    fn trace(&self, scene: &Scene, ray: Ray) -> Vec3;
}

pub struct NormalTracer;

impl Tracer for NormalTracer {
    fn trace(&self, scene: &Scene, ray: Ray) -> Vec3 {
        match scene.shape.cast(ray, None) {
            None => Vec3::ZERO,
            Some(hit) => (scene.shape.normal(hit.point) + Vec3::ONE) / 2.0,
        }
    }
}

pub struct PathTracer {
    samples: usize,
    bounces: usize,
}

impl PathTracer {
    fn trace_rec(&self, scene: &Scene, ray: Ray, bounces: usize) -> Vec3 {
        if bounces == 0 {
            return Vec3::ZERO;
        }
        match scene.shape.cast(ray, None) {
            None => Vec3::ZERO,
            Some(hit) => {
                let normal = scene.shape.normal(hit.point);

                // direct illumination by light
                let direct: Vec3 = scene
                    .lights
                    .iter()
                    .filter_map(|light| {
                        let light_at = light.illuminate(hit.point);
                        let shadow_ray = Ray::new(hit.point, -light_at.direction);
                        match scene.shape.cast(shadow_ray, Some(light_at.distance)) {
                            None => {
                                Some(light_at.intensity * normal.dot(-light_at.direction).max(0.0))
                            }
                            Some(_) => None,
                        }
                    })
                    .sum();
                let direct = hit.material.reflectance * direct / PI;

                // global illumination
                let indirect = if self.samples > 0 {
                    let coords = create_coords(normal);
                    let pdf = 1.0 / (2.0 * PI); // uniform hemesphere PDF
                    let brdf = hit.material.reflectance / PI;
                    let samples_sum: Vec3 = (0..self.samples)
                        .map(|_| {
                            let sample = &coords * uniform_hemisphere_sample(); // sample in world coordinates
                            let sample_ray = Ray::new(hit.point, sample);
                            let cos_theta = sample.dot(normal);
                            let incoming = self.trace_rec(scene, sample_ray, bounces - 1);
                            brdf * incoming * cos_theta / pdf
                        })
                        .sum();
                    samples_sum / self.samples as Scalar
                } else {
                    Vec3::ZERO
                };

                direct + indirect + hit.material.emittance
            }
        }
    }
}

impl Tracer for PathTracer {
    fn trace(&self, scene: &Scene, ray: Ray) -> Vec3 {
        self.trace_rec(scene, ray, self.bounces)
    }
}

// -----------------------------------------------------------------------------
// Utils
// -----------------------------------------------------------------------------
/// Create column matrix of euclidian coordinates system with y-axis along normal.
fn create_coords(normal: Vec3) -> Mat3 {
    let Vec3([ix, iy, iz]) = normal;
    let Vec3([jx, jy, jz]) = if ix.abs() > iy.abs() {
        Vec3([iz, 0.0, -ix]).normalize()
    } else {
        Vec3([0.0, -iz, iy]).normalize()
    };
    let Vec3([kx, ky, kz]) = normal.cross(Vec3([jx, jy, jz]));
    Mat3([kx, ix, jx, ky, iy, jy, kz, iz, jz])
}

/// Generate random vector in hemisphere around y-axis
fn uniform_hemisphere_sample() -> Vec3 {
    let r1 = rand::random::<Scalar>();
    let r2 = rand::random::<Scalar>();
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    let sin_theta = (1.0 - r1 * r1).sqrt();
    let phi = 2.0 * PI * r2;
    let x = sin_theta * phi.cos();
    let z = sin_theta * phi.sin();
    Vec3([x, r1, z])
}

/// Calculate direction of ray given field of view in radians, and the size,
/// and the coordinates on the canvas
fn ray_direction(field_of_view: Scalar, width: u32, height: u32, x: u32, y: u32) -> Vec3 {
    let x = x as Scalar - width as Scalar / 2.0;
    let y = y as Scalar - height as Scalar / 2.0;
    let z = height as Scalar / (field_of_view / 2.0).tan();
    return Vec3([x, -y, -z]).normalize();
}

fn image_from_fn<F>(width: u32, height: u32, f: F) -> RgbImage
where
    F: Fn(u32, u32) -> Rgb<u8> + Send + Sync,
{
    use rayon::prelude::*;

    let coords: Vec<_> = (0..height)
        .flat_map(move |y| (0..width).map(move |x| (x, y)))
        .collect();
    let storage: Vec<_> = coords // render in parallel iterator
        .into_par_iter()
        .map(|(x, y)| f(x, y))
        // flat map pixels in serial iterator
        .collect::<Vec<_>>()
        .iter()
        .flat_map(|pixel| &pixel.data)
        .cloned()
        .collect();
    ImageBuffer::from_vec(width, height, storage).expect("ImageBuffer::from_raw failed")
}

fn rad(deg: Scalar) -> Scalar {
    deg / 180.0 * PI
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------
fn scene() -> Scene {
    // Shape
    let m0 = Material::new(Vec3([184.0, 187.0, 38.0]) / 255.0); // geen-bold
    let m1 = Material::new(Vec3([152.0, 151.0, 26.0]) / 255.0); // green
    let m2 = Material::new(Vec3([134.0, 98.0, 177.0]) / 255.0); // magenta
    let m3 = Material::new(Vec3::ZERO).with_emittance(Vec3::ONE * 15.0);

    let shape = Box::new(m0.clone(), Vec3([1.5; 3]))
        .intersect(Sphere::new(m1.clone(), Vec3::ZERO, 1.0))
        .diff(
            Cylinder::new(m2.clone(), 2.0, 0.5)
                .union(Cylinder::new(m2.clone(), 2.0, 0.5).rot(Mat3::rot_x(rad(90.0))))
                .union(Cylinder::new(m2.clone(), 2.0, 0.5).rot(Mat3::rot_y(rad(90.0)))),
        )
        .rot(&Mat3::rot_x(rad(-35.0)) * &Mat3::rot_y(rad(-30.0)))
        .scale(1.5);

    let tor = Torus::new(m3, 1.0, 0.3)
        .rot(&Mat3::rot_x(rad(45.)) * &Mat3::rot_y(rad(-45.0)))
        .translate(Vec3([1.0, -1.0, -1.0]) * 2.5);

    let room = Box::new(Material::new(Vec3([1.0; 3])), Vec3([10.0; 3])).invert();

    let shape = room.union(tor.union(shape));

    // Lights
    let l0 = std::boxed::Box::new(PointLight {
        color: Vec3::ONE,
        intensity: 200.0,
        position: Vec3([1.5, -0.5, 3.0]),
    });
    let l1 = std::boxed::Box::new(PointLight {
        color: Vec3::ONE,
        intensity: 200.0,
        position: Vec3([0.0, 3.0, 0.0]),
    });

    Scene {
        shape: std::boxed::Box::new(shape),
        lights: vec![l0, l1],
    }
}

fn main() {
    let origin = Vec3([0.0, 0.0, 4.9]);
    let scene = scene();

    let tracer = PathTracer {
        samples: 32,
        bounces: 2,
    };
    // let tracer = NormalTracer;

    let (width, height) = (512, 512);
    let image = image_from_fn(width, height, |x, y| {
        let ray = Ray::new(origin, ray_direction(rad(120.0), width, height, x, y));
        let color = tracer.trace(&scene, ray);
        // tone mapping
        Rgb::<u8>::from(color / (color + Vec3::ONE))
    });

    image
        .save("path-tracer-output.png")
        .expect("failed to save image")
}
