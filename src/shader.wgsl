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
    out.clip_position = camera.view_proj * modelUbo.transform * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

// Fragment shader

fn sdf( p: vec3<f32> ) -> f32 {
    return length( p ) - 1.0;
}

fn raycast( position: vec3<f32>, direction: vec3<f32> ) -> f32 {

    var t = 0.0;
    for( var i = 0; i < 500; i++ ) {
        let d = sdf( position + direction * t );
        if( d < 0.001 ) {
            return t;
        }
        t += d;
    }

    return -1.0;
}

@fragment
fn fs_main( in: VertexOutput ) -> @location(0) vec4<f32> {
    let q = in.clip_position.xy;

    let uv = in.clip_position.xy * 2.0f - 1.0f;

    let near = 1.0f;
    let far = 1000.0f;
    let worldpos4 = ( ( camera.view_proj * vec4( uv, -1.0, 1.0 ) ) );
    let worldpos = worldpos4.xyz / worldpos4.w * near;
    let worlddir = normalize( ( camera.view_proj * vec4( -uv * ( far - near ), far + near, far - near ) ).xyz);

    let origin = vec3<f32>( 0.0, 0.0, 0.0 );
    let direction = normalize( vec3<f32>( q, 1.0 ) );
    let t = raycast( origin, direction );
    return vec4<f32>( worldpos.xyz / 20.0, 1.0);
//    if( t > 0.0 ) {
//        return vec4<f32>( 1.0, 0.0, 0.0, 1.0);
//    }
//
//    return vec4<f32>(in.color * t, 1.0);
}