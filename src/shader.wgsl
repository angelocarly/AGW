// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

struct ModelUniform {
    transform: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> modelUbo: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = modelUbo.transform * camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

// Fragment shader

@fragment
fn fs_main( in: VertexOutput ) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}