//! The vertex shader of a mesh fallback (hosts without mesh shaders): it
//! draws the records a mesh program run as compute wrote (see
//! [`super::lower::Record`]). A non-indexed draw of `3 * max_primitives`
//! vertices per instance, one instance per group: vertex `v` of instance
//! `g` is corner `v % 3` of primitive `v / 3` of group `g`'s record. A
//! culled primitive's indices are all 0, so its triangle has no area.
//!
//! The outputs are the pixel shader's inputs as its signature declares
//! them (same elements, rows and columns), so the two link; each takes the
//! mesh output (per vertex) or primitive output (the same for the
//! triangle's three corners) of that semantic, or 0 if the program writes
//! none. Without a pixel shader only the position is written.

use super::bitcode::{Attr, Bin, Cast, Term, V};
use super::container::SigElement;
use super::lower::Record;
use super::shader::Shader;

/// DXIL semantic kinds used here.
const VERTEX_ID: u8 = 1;
const INSTANCE_ID: u8 = 2;
const POSITION: u8 = 3;
/// Clip and cull distances.
const CLIP: u8 = 6;
const CULL: u8 = 7;

/// Where an output takes its value from.
#[derive(Clone, Copy)]
enum Source {
    Vertex(u32),
    Primitive(u32),
    Zero,
}

fn input(name: &str, kind: u8, row: u32) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 5,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }
}

/// The interpolation a vertex output declares (as the emitter's defaults).
fn interp(e: &SigElement) -> u8 {
    match (e.kind, e.comp) {
        (POSITION, _) => 4,
        (0 | CLIP | CULL, 9) => 2,
        (0, _) => 1,
        _ => 0,
    }
}

/// Builds the vertex shader for a mesh program with these vertex and
/// primitive outputs, whose pipeline's pixel shader reads `ps_inputs`.
pub(crate) fn vertex_shader(
    record: &Record,
    outputs: &[SigElement],
    prims: &[SigElement],
    ps_inputs: &[SigElement],
) -> Result<Vec<u8>, &'static str> {
    let same = |a: &SigElement, b: &SigElement| {
        a.kind == b.kind && a.index == b.index && a.name.eq_ignore_ascii_case(&b.name)
    };
    let position = outputs
        .iter()
        .position(|o| o.kind == POSITION)
        .ok_or("a mesh program without a position output")?;
    let mut elements: Vec<(SigElement, Source)> = Vec::new();
    for e in ps_inputs {
        let source = match e.kind {
            POSITION => Source::Vertex(position as u32),
            0 | CLIP | CULL => {
                if let Some(k) = outputs.iter().position(|o| same(o, e)) {
                    Source::Vertex(k as u32)
                } else if let Some(k) = prims.iter().position(|o| same(o, e)) {
                    Source::Primitive(k as u32)
                } else {
                    Source::Zero
                }
            }
            // Values the rasterizer makes (front face, sample index...).
            _ => continue,
        };
        elements.push((e.clone(), source));
    }
    // The position and clip or cull distances are written even if the
    // pixel shader does not read them, in rows past its own.
    let mut next_row = ps_inputs
        .iter()
        .filter_map(|e| e.row)
        .max()
        .map_or(0, |r| r + 1);
    for (k, o) in outputs.iter().enumerate() {
        if matches!(o.kind, POSITION | CLIP | CULL) && !elements.iter().any(|(e, _)| same(e, o)) {
            let mut e = o.clone();
            e.row = Some(next_row);
            next_row += 1;
            elements.push((e, Source::Vertex(k as u32)));
        }
    }

    let mut s = Shader::new(false);
    s.inputs = vec![
        input("SV_VertexID", VERTEX_ID, 0),
        input("SV_InstanceID", INSTANCE_ID, 1),
    ];
    s.outputs = elements
        .iter()
        .map(|(e, _)| SigElement {
            interp: interp(e),
            usage: ((1u16 << e.cols) - 1) as u8,
            half: false,
            ..e.clone()
        })
        .collect();

    let b = s.entry;
    let i32t = s.t.i32;
    let load_input = |s: &mut Shader, id: u32| {
        let args = [s.m.i32(id), s.m.i32(0), s.m.i8(0), s.m.undef(i32t)];
        s.op(b, "loadInput.i32", Attr::ReadNone, i32t, 4, &args)
    };
    let vid = load_input(&mut s, 0);
    let group = load_input(&mut s, 1);
    let h = s.record_srv_handle();
    let k = |s: &mut Shader, v: u32| s.m.i32(v);
    let three = k(&mut s, 3);
    let prim = s.m.bin(b, Bin::UDiv, vid, three);
    let corner = s.m.bin(b, Bin::URem, vid, three);
    let rec = k(&mut s, record.dwords());
    let base = s.m.bin(b, Bin::Mul, group, rec);
    let load = |s: &mut Shader, dword: V| -> V {
        let four = s.m.i32(4);
        let off = s.m.bin(b, Bin::Mul, dword, four);
        let (u, mask, al) = (s.m.undef(i32t), s.m.i8(1), s.m.i32(4));
        let res_ret = s.t.res_ret;
        let r = s.op(
            b,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res_ret,
            139,
            &[h, off, u, mask, al],
        );
        let x = s.m.extract(b, r, 0);
        s.m.cast(b, Cast::Bitcast, x, i32t)
    };
    // The counts the program requested (dword 0), else the maxima.
    let (vertices, primitives) = if record.counts {
        let counts = load(&mut s, base);
        let low = k(&mut s, 0x1FF);
        let twelve = k(&mut s, 12);
        let v = s.m.bin(b, Bin::And, counts, low);
        let p = s.m.bin(b, Bin::LShr, counts, twelve);
        (v, s.m.bin(b, Bin::And, p, low))
    } else {
        (
            k(&mut s, record.max_vertices),
            k(&mut s, record.max_primitives),
        )
    };
    // The corner's vertex: its 10-bit index in the packed primitive; a
    // primitive past the count is culled (indices 0), as on the console.
    let first = k(&mut s, Record::INDICES_AT);
    let packed_at = s.m.bin(b, Bin::Add, base, prim);
    let packed_at = s.m.bin(b, Bin::Add, packed_at, first);
    let packed = load(&mut s, packed_at);
    let counted = s.m.cmp(b, super::bitcode::pred::IULT, prim, primitives);
    let zero = k(&mut s, 0);
    let packed = s.m.select(b, counted, packed, zero);
    let ten = k(&mut s, 10);
    let shift = s.m.bin(b, Bin::Mul, corner, ten);
    let index = s.m.bin(b, Bin::LShr, packed, shift);
    let mask = k(&mut s, 0x3FF);
    let index = s.m.bin(b, Bin::And, index, mask);
    // An index past the program's vertices (a slot no lane wrote) reads
    // vertex 0: root views have no bounds, so it must stay in the record.
    let max_v = k(&mut s, record.max_vertices);
    let inside = s.m.cmp(b, super::bitcode::pred::IULT, index, max_v);
    let counted = s.m.cmp(b, super::bitcode::pred::IULT, index, vertices);
    let inside = s.m.bin(b, Bin::And, inside, counted);
    let index = s.m.select(b, inside, index, zero);
    let at = |s: &mut Shader, first: u32, per: u32, which: V| -> V {
        let n = s.m.i32(per * 4);
        let x = s.m.bin(b, Bin::Mul, which, n);
        let x = s.m.bin(b, Bin::Add, base, x);
        let f = s.m.i32(first);
        s.m.bin(b, Bin::Add, x, f)
    };
    let vertex = at(&mut s, record.vertices_at(), record.outputs, index);
    let primitive = at(&mut s, record.prims_at(), record.prims, prim);

    let void = s.t.void;
    let f32t = s.t.f32;
    for (id, (e, source)) in elements.iter().enumerate() {
        // Columns relative to the element, as the record and DXIL count them;
        // ones the mesh output lacks read 0.
        for col in 0..e.cols {
            let x = match *source {
                Source::Vertex(el) if col < outputs[el as usize].cols => {
                    let d = k(&mut s, el * 4 + u32::from(col));
                    let d = s.m.bin(b, Bin::Add, vertex, d);
                    load(&mut s, d)
                }
                Source::Primitive(el) if col < prims[el as usize].cols => {
                    let d = k(&mut s, el * 4 + u32::from(col));
                    let d = s.m.bin(b, Bin::Add, primitive, d);
                    load(&mut s, d)
                }
                _ => k(&mut s, 0),
            };
            let (suffix, x) = if e.comp == 9 {
                (".f32", s.m.cast(b, Cast::Bitcast, x, f32t))
            } else {
                (".i32", x)
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), x];
            s.op(
                b,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                void,
                5,
                &args,
            );
        }
    }
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}
