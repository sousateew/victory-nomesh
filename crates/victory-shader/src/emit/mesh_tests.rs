//! End-to-end tests of synthetic GFX10 mesh programs: assembled here, lowered
//! to DXIL, validated by `dxil.dll` when present, and drawn on every SM 6.6
//! host with mesh shaders (WARP has them; GPUs without them skip).
//!
//! Each program also runs through the mesh fallback (compute plus a vertex
//! shader, for hosts without mesh shaders) on the hardware device and on
//! WARP, and the hardware fallback must draw the pixels WARP's native mesh
//! draw does.
//!
//! Unlike `diff_tests`, these need no console compiler, so the mesh path
//! (thread index, position and primitive exports, culling) is covered for
//! every contributor.

use windows::Win32::Graphics::Direct3D12::*;

use super::container::SigElement;
use super::host::{Draw, Fallback, Host};
use super::{HostShader, MeshFallback, lower, validator};
use crate::native::{HwStage, MeshIo, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constants as source operands.
const ZERO: u32 = 0x80;
const ONE_F: u32 = 0xF2;
const MINUS_ONE_F: u32 = 0xF3;
const MINUS_ONE: u32 = 0xC1;
const LITERAL: u32 = 0xFF;

/// VOP1 `v_mov_b32 vdst, src0`.
const fn v_mov(vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | 0x01 << 9 | src0
}

/// VOPC `v_cmp_eq_u32 vcc, src0, vsrc1`.
const fn v_cmp_eq_u32(src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | 0xC2 << 17 | vsrc1 << 9 | src0
}

/// VOP2 `v_cndmask_b32 vdst, src0, vsrc1, vcc` (vcc ? vsrc1 : src0).
const fn v_cndmask(vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    0x01 << 25 | vdst << 17 | vsrc1 << 9 | src0
}

/// VOP3 `v_mbcnt_lo_u32_b32 vdst, -1, 0`: the lane index.
const fn lane_index(vdst: u32) -> [u32; 2] {
    [0xD400_0000 | 0x365 << 16 | vdst, MINUS_ONE | ZERO << 9]
}

/// EXP of four VGPRs (`en` channel mask) to `target`, with `done`.
const fn exp(target: u32, en: u32, v: [u32; 4]) -> [u32; 2] {
    [
        0xF800_0000 | 1 << 11 | target << 4 | en,
        v[0] | v[1] << 8 | v[2] << 16 | v[3] << 24,
    ]
}

/// One triangle per group: lane n exports vertex n of (-1, 1), (1, 1),
/// (-1, -1) (the target's upper-left half) and lane 0 the primitive
/// `primitive` (10-bit indices, bit 31 culls); lanes 1 and 2 cull theirs.
fn triangle_program(primitive: u32) -> Vec<u32> {
    let mut w = Vec::new();
    w.extend(lane_index(1));
    // Primitive: lane 0 exports `primitive`, the others a culled one.
    w.push(v_cmp_eq_u32(ZERO, 1));
    w.extend([v_mov(8, LITERAL), primitive]);
    w.extend([v_cndmask(7, LITERAL, 8), 0x8000_0000]);
    w.extend(exp(20, 0x1, [7, 0, 0, 0]));
    // x: 1 in lane 1, else -1.
    w.push(v_mov(3, ONE_F));
    w.push(v_cmp_eq_u32(0x81, 1));
    w.push(v_cndmask(2, MINUS_ONE_F, 3));
    // y: -1 in lane 2, else 1.
    w.push(v_mov(5, MINUS_ONE_F));
    w.push(v_cmp_eq_u32(0x82, 1));
    w.push(v_cndmask(4, ONE_F, 5));
    w.push(v_mov(6, ZERO));
    w.extend(exp(12, 0xF, [2, 4, 6, 3]));
    w.push(S_ENDPGM);
    w
}

/// `s_mov_b32 m0, counts` then `s_sendmsg` GS_ALLOC_REQ: the output
/// counts (vertices | primitives << 12), before `program`'s code.
fn with_counts(vertices: u32, primitives: u32, program: Vec<u32>) -> Vec<u32> {
    let mut w = vec![0xBEFC_03FF, vertices | primitives << 12, 0xBF90_0009];
    w.extend(program);
    w
}

fn element(name: &str, kind: u8, interp: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    }
}

/// Lowers a mesh program of three lanes, three vertices and three
/// primitives that exports only a position (as compute writing records
/// with `mesh_compute`).
fn lower_mesh(words: &[u32], mesh_compute: bool) -> (HostShader, lower::Lowered) {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: 3,
            max_vertices: 3,
            max_primitives: 3,
            topology: 2,
            lds_bytes: 0,
        }),
        code,
    };
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: vec![element("SV_Position", 3, 4)],
        prims: Vec::new(),
        mesh_stride: 3,
        mesh_compute,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Ms,
        dxil: lowered.dxil.clone(),
        user_data: lowered.user_data.clone(),
        samplers: lowered.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs.clone(),
        root_srvs: lowered.root_srvs.clone(),
        root_uavs: lowered.root_uavs.clone(),
        root_indexed_views: Default::default(),
    };
    validate(&shader);
    (shader, lowered)
}

fn translate_mesh(words: &[u32]) -> HostShader {
    lower_mesh(words, false).0
}

/// The program's mesh fallback for [`ps_writing_one`] (which reads no
/// inputs), both halves validated.
fn translate_fallback(words: &[u32]) -> MeshFallback {
    let (compute, lowered) = lower_mesh(words, true);
    let f = super::mesh_fallback_of(compute, &lowered, &[]).expect("mesh fallback");
    validate(&vertex_of(&f));
    f
}

fn vertex_of(f: &MeshFallback) -> HostShader {
    HostShader {
        stage: HwStage::Vs,
        dxil: f.vertex.clone(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Default::default(),
    }
}

fn validate(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
}

/// A pixel shader that writes 1 to every channel of target 0.
fn ps_writing_one() -> HostShader {
    use super::bitcode::{Attr, Term};
    let mut s = super::shader::Shader::new(true);
    let b = s.entry;
    for col in 0..4 {
        let args = [
            s.m.i32(0),
            s.m.i32(0),
            s.m.i8(col),
            s.m.f32(1.0f32.to_bits()),
        ];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    let mut target = element("SV_Target", 16, 0);
    target.usage = 0xF;
    s.outputs = vec![target];
    s.m.terminate(b, Term::Ret);
    let shader = HostShader {
        stage: HwStage::Ps,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Default::default(),
    };
    validate(&shader);
    shader
}

fn mesh_shaders(host: &Host) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    // SAFETY: the query writes the structure of the given size.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
}

const SIZE: u32 = 8;

/// Draws one group of `ms` on every SM 6.6 host with mesh shaders and
/// returns, per host that ran, which pixels were written (row-major).
fn draw_on_mesh_hosts(ms: &HostShader) -> Vec<(&'static str, Vec<bool>)> {
    let ps = ps_writing_one();
    let mut ran = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let host = match Host::new(warp) {
            Ok(h) if mesh_shaders(&h) => h,
            Ok(_) => {
                eprintln!("{label}: no mesh shaders; skipped");
                continue;
            }
            Err(e) => {
                eprintln!("{label}: {e}; skipped");
                continue;
            }
        };
        let (root, pso) = host
            .mesh_pipeline(ms, &ps)
            .unwrap_or_else(|e| panic!("{label}: mesh pipeline refused: {e}"));
        let texels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (SIZE, SIZE),
                    mesh_groups: 1,
                    fallback: None,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
        let written: Vec<bool> = texels.iter().map(|t| t[0] == 1.0).collect();
        eprintln!(
            "{label}: drew, {} of {} pixels written",
            written.iter().filter(|&&w| w).count(),
            written.len()
        );
        ran.push((label, written));
    }
    if ran.is_empty() {
        eprintln!("no SM 6.6 host with mesh shaders; nothing drawn");
    }
    ran
}

/// Draws one group through the fallback `f` on the hardware device and on
/// WARP, as a host without mesh shaders would.
fn draw_fallback(f: &MeshFallback) -> Vec<(&'static str, Vec<bool>)> {
    let ps = ps_writing_one();
    let vs = vertex_of(f);
    let mut ran = Vec::new();
    for (label, warp) in [("hardware fallback", false), ("warp fallback", true)] {
        let host = match Host::new(warp) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{label}: {e}; skipped");
                continue;
            }
        };
        let (croot, uav) = host
            .root_signature_with_record(&[(&f.compute, D3D12_SHADER_VISIBILITY_ALL)], false)
            .unwrap_or_else(|e| panic!("{label}: compute root signature refused: {e}"));
        let cpso = host
            .compute_pipeline_with(&croot, &f.compute.dxil)
            .unwrap_or_else(|e| panic!("{label}: compute pipeline refused: {e}"));
        let (root, srv) = host
            .root_signature_with_record(
                &[
                    (&vs, D3D12_SHADER_VISIBILITY_VERTEX),
                    (&ps, D3D12_SHADER_VISIBILITY_PIXEL),
                ],
                true,
            )
            .unwrap_or_else(|e| panic!("{label}: root signature refused: {e}"));
        let pso = host
            .pipeline_with(&root, &vs.dxil, &ps.dxil)
            .unwrap_or_else(|e| panic!("{label}: pipeline refused: {e}"));
        let texels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (SIZE, SIZE),
                    mesh_groups: 1,
                    fallback: Some(Fallback {
                        root: &croot,
                        pso: &cpso,
                        uav,
                        srv,
                        record_bytes: f.record_bytes,
                        vertices: 3 * f.max_primitives,
                    }),
                },
            )
            .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
        let written: Vec<bool> = texels.iter().map(|t| t[0] == 1.0).collect();
        eprintln!(
            "{label}: drew, {} of {} pixels written",
            written.iter().filter(|&&w| w).count(),
            written.len()
        );
        ran.push((label, written));
    }
    ran
}

/// Draws the program natively on mesh hosts and through its fallback; the
/// hardware fallback must match WARP's native draw. Every result is
/// returned for the caller's own checks.
fn draw_everywhere(words: &[u32]) -> Vec<(&'static str, Vec<bool>)> {
    let mut ran = draw_on_mesh_hosts(&translate_mesh(words));
    ran.extend(draw_fallback(&translate_fallback(words)));
    let find = |label| ran.iter().find(|r| r.0 == label).map(|r| &r.1);
    match (find("hardware fallback"), find("warp")) {
        (Some(fallback), Some(native)) => {
            assert_eq!(
                fallback, native,
                "hardware fallback differs from WARP's mesh draw"
            )
        }
        _ => eprintln!("no hardware fallback or WARP mesh draw; not compared"),
    }
    ran
}

#[test]
fn a_mesh_triangle_covers_its_half_of_the_target() {
    for (label, written) in draw_everywhere(&triangle_program(1 << 10 | 2 << 20)) {
        for y in 0..SIZE {
            for x in 0..SIZE {
                // Centers on the diagonal x + y = 7 lie on the edge.
                if x + y == SIZE - 1 {
                    continue;
                }
                let inside = x + y < SIZE - 1;
                assert_eq!(
                    written[(y * SIZE + x) as usize],
                    inside,
                    "{label}: pixel ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn a_culled_mesh_primitive_draws_nothing() {
    for (label, written) in draw_everywhere(&triangle_program(0x8000_0000 | 1 << 10 | 2 << 20)) {
        assert!(
            written.iter().all(|&w| !w),
            "{label}: a culled primitive drew"
        );
    }
}

#[test]
fn mesh_vertex_indices_pick_the_exported_vertices() {
    // Vertices 0, 1, 1: no area, nothing drawn; a lowering that ignored the
    // exported indices (vertex n for corner n) would draw the half.
    for (label, written) in draw_everywhere(&triangle_program(1 << 10 | 1 << 20)) {
        assert!(
            written.iter().all(|&w| !w),
            "{label}: a zero-area primitive drew"
        );
    }
}

#[test]
fn a_mesh_fallback_draws_only_the_requested_primitives() {
    // The program exports primitive 0 but requests none: the console draws
    // nothing (the native path, which ignores the counts, is not compared).
    let words = with_counts(3, 0, triangle_program(1 << 10 | 2 << 20));
    for (label, written) in draw_fallback(&translate_fallback(&words)) {
        assert!(
            written.iter().all(|&w| !w),
            "{label}: a primitive past the count drew"
        );
    }
    // Requesting it draws it.
    let words = with_counts(3, 1, triangle_program(1 << 10 | 2 << 20));
    for (label, written) in draw_fallback(&translate_fallback(&words)) {
        assert_eq!(
            written.iter().filter(|&&w| w).count(),
            28,
            "{label}: the requested primitive"
        );
    }
}

#[test]
fn a_fallback_vertex_shader_feeds_elements_off_column_zero() {
    // A pixel input packed into columns 2-3 of its row (and one wider than
    // the mesh output): the vertex shader must store columns relative to
    // the element, or the validator refuses it.
    let mut uv = element("TEXCOORD", 0, 2);
    (uv.row, uv.col, uv.cols) = (Some(1), 2, 2);
    let mut wide = element("TEXCOORD", 0, 2);
    (wide.index, wide.row, wide.cols) = (1, Some(2), 4);
    let mut narrow = wide.clone();
    narrow.cols = 2;
    let outputs = vec![element("SV_Position", 3, 4), uv.clone(), narrow];
    let record = lower::Record {
        max_vertices: 3,
        max_primitives: 1,
        outputs: outputs.len() as u32,
        prims: 0,
        counts: false,
    };
    let ps_inputs = vec![element("SV_Position", 3, 4), uv, wide];
    let vs =
        super::fallback::vertex_shader(&record, &outputs, &[], &ps_inputs).expect("vertex shader");
    let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref())
    else {
        eprintln!("no dxil.dll; not validated");
        return;
    };
    let mut copy = vs;
    v.validate(&mut copy).expect("DXIL validation failed");
}
