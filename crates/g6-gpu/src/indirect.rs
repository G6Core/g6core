//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

pub(crate) mod batch;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: one dispatch, written as
    // SPLIT records of at most 65,535 groups in x each.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool split = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 asked = src_args.Load3(src_at);
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // In parts: one dimension, and not the garbage of an argument
        // nothing wrote.
        if (split && g.y <= 1u && g.z <= 1u && asked.x <= SPLIT * 65535u) {
            g.x = asked.x;
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (split) {
            for (uint part = 0; part < SPLIT; part++) {
                uint before = part * 65535u;
                uint x = groups.x > before ? min(groups.x - before, 65535u) : 0u;
                dst_args.Store3(dst_at + part * 12u, uint3(x, groups.yz));
            }
            return;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_from(device, SOURCE)
}

/// The pass over many records ([`batch`]), made once.
fn batch_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let made = host().ok_or_else(windows::core::Error::empty).and_then(|h| make_from(&h.device, batch::SOURCE));
            match made {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no grouped argument clamp: each record is bounded on its own");
                    None
                }
            }
        })
        .as_ref()
}

/// Records, in `list`, a pass that bounds the records the returned table is
/// given, now and while the list is recorded ([`batch`]), from `source` (a
/// title buffer copy and its number) and the copies read before it
/// (`recent`). Returns the table and the buffer the bounded records lie in,
/// which is left in `INDIRECT_ARGUMENT`. Replaces the list's compute root
/// signature and pipeline. `None` if there is no such pass.
pub(crate) fn open_batch(
    list: &HostList,
    source: (u64, &ID3D12Resource),
    recent: &[(u64, ID3D12Resource)],
) -> Option<(batch::Table, ID3D12Resource)> {
    let (c, budget) = (batch_clamp()?, list.budget()?);
    let (jobs, at) = list.scratch(batch::TABLE)?;
    let (dst, before) = list.batch_buffer(batch::BUFFER)?;
    let address = |b: &ID3D12Resource| {
        // SAFETY: a live buffer.
        Some(unsafe { b.GetGPUVirtualAddress() }).filter(|at| at.is_multiple_of(4))
    };
    // A source there is none of is the first again, which no job names.
    let mut ids = [0u64; batch::SOURCES];
    let mut sources = [address(source.1)?; batch::SOURCES];
    ids[0] = source.0;
    let others = recent.iter().filter(|r| r.0 != source.0).take(batch::SOURCES - 1);
    for (i, (id, buffer)) in others.enumerate() {
        if let Some(at) = address(buffer) {
            // Bound, and read only if a job names it.
            list.keep(buffer);
            (ids[i + 1], sources[i + 1]) = (*id, at);
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    batch::record(c, &list.list, (&dst, before), budget, sources, jobs);
    // SAFETY: the list's own mapped upload memory of that many bytes,
    // 256-byte aligned, until the list is reset.
    Some((unsafe { batch::Table::new(at, ids) }, dst))
}

fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    record_clamp(c, &list.list, &dst, budget, src, count, spec).then_some(dst)
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    record_clamp_as(c, &list.list, &dst, budget, src, None, spec, AHEAD).then_some(dst)
}

/// The parts [`bound_split`] makes of a dispatch, the most groups of each,
/// and the bytes between them.
pub(crate) const SPLIT: u32 = 2;
pub(crate) const PART: u32 = 65_535;
pub(crate) const PART_BYTES: u64 = 12;

/// [`bound`] for one dispatch that a program with a group base runs
/// (`g6_shader::emit::USER_GROUP_BASE`): [`SPLIT`] records follow each
/// other in the scratch buffer, of which the first has the first [`PART`]
/// groups in x and the next the ones after (empty if there are none). The
/// console runs 115,200 groups in one dispatch (the title's final
/// gather of indirect light: 8 for each 16x16 tile of 2560x1440); clamped
/// to what the host runs, the tiles at the bottom of the screen kept what
/// they showed before.
pub(crate) fn bound_split(list: &HostList, src: (&ID3D12Resource, u64)) -> Option<ID3D12Resource> {
    let spec = (2, 12, 1);
    validate_layout(spec, src.1, None)?;
    let (c, budget) = (clamp()?, list.budget()?);
    let dst = list.indirect_scratch(u64::from(SPLIT) * PART_BYTES)?;
    record_clamp_as(c, &list.list, &dst, budget, src, None, spec, IN_PARTS).then_some(dst)
}

/// Bits of the clamp's `has_count`.
const AHEAD: u32 = 2;
const IN_PARTS: u32 = 4;

fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> bool {
    record_clamp_as(c, l, dst, budget, src, count, spec, 0)
}

/// [`record_clamp`]; `how`: [`AHEAD`], the records are a window clamped
/// ahead of their draws ([`bound_ahead`]); [`IN_PARTS`], one dispatch in
/// parts ([`bound_split`]).
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    how: u32,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [kind, stride, records, args_offset, has_count | how, count_offset];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    // Diagnostics: counted; `G6_BARRIER_TEST=indirect` leaves the two
    // waits for every earlier shader write out (the state changes stay).
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let first = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ),
    ];
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let skip = usize::from(!waits);
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[skip..]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        l.ResourceBarrier(&second[skip..]);
    }
    true
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    done_on(&list.list, scratch);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// What the pass that bounds indirect commands costs (`G6_CLAMP_TIMING=1`,
/// diagnostics builds). It measures only: the commands recorded for the
/// picture are the same with it on. Off, nothing is read or counted.
///
/// Processor time is summed over the recording threads and is not time on
/// the card. The card's time is a timing run's (`G6_GPU_TIMING` with
/// `G6_D3D12_DRED`): the list notes `clamp` before a pass and `clamped`
/// after it, so that run's times by command show the pass apart (`clamp`),
/// the state set again after it (`clamped`), and the commands that followed
/// a pass (`indirect+pass`) apart from those that needed none
/// (`indirect+window`). Those two are different commands, not the same ones
/// with and without a pass.
#[cfg(feature = "diagnostics")]
pub(crate) mod cost {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    pub(crate) fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("G6_CLAMP_TIMING").is_ok_and(|v| v == "1"))
    }

    /// What was timed.
    #[derive(Debug, Clone, Copy)]
    pub(crate) enum Part {
        /// Recording a pass for the records of one command.
        Single,
        /// Recording a pass for a window or a group of records, which the
        /// commands that follow read too (for a group the records covered are
        /// counted as one: the rest join after).
        Window,
        /// Recording a pass that splits a large dispatch in parts.
        Split,
        /// Preparing the command after a pass: all of it, most of which it
        /// needs anyway, with its pipeline and roots set again.
        AfterPass,
        /// The same for a command that needed no pass.
        InWindow,
        /// Returning the pass's buffer after a command.
        Done,
    }
    const PARTS: usize = 6;
    static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
    static TIMES: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
    /// Records the passes covered.
    static RECORDS: AtomicU64 = AtomicU64::new(0);
    /// Passes by their command: draw, indexed draw, dispatch, mesh.
    static KINDS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

    /// A command's kind (the numbers of `crate::draw`), as a word.
    pub(crate) fn kind_name(kind: u32) -> &'static str {
        ["draw", "indexed", "dispatch", "mesh"][kind_at(kind)]
    }

    fn kind_at(kind: u32) -> usize {
        match kind {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        }
    }

    /// `part` took from `since` to now.
    pub(crate) fn took(part: Part, since: Instant) {
        NS[part as usize].fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
        TIMES[part as usize].fetch_add(1, Relaxed);
    }

    /// A pass for `records` records of a command of `kind` was recorded.
    pub(crate) fn pass(part: Part, kind: u32, records: u32, since: Instant) {
        took(part, since);
        RECORDS.fetch_add(u64::from(records), Relaxed);
        KINDS[kind_at(kind)].fetch_add(1, Relaxed);
    }

    /// For each part: how many, and the nanoseconds, since the last taking.
    pub(super) fn take() -> ([(u64, u64); PARTS], u64, [u64; 4]) {
        (
            std::array::from_fn(|i| (TIMES[i].swap(0, Relaxed), NS[i].swap(0, Relaxed))),
            RECORDS.swap(0, Relaxed),
            std::array::from_fn(|i| KINDS[i].swap(0, Relaxed)),
        )
    }

    /// The line for the frame-rate report.
    pub(crate) fn log(frames: f64) {
        if !on() {
            return;
        }
        let (parts, records, kinds) = take();
        let [single, window, split, after, within, done] = parts.map(|(n, ns)| {
            (n as f64 / frames, ns as f64 / 1e6 / frames, if n == 0 { 0.0 } else { ns as f64 / 1e3 / n as f64 })
        });
        let kinds = kinds.map(|n| n as f64 / frames);
        let passes = single.0 + window.0 + split.0;
        tracing::info!(
            "  the clamp pass per frame (G6_CLAMP_TIMING): {passes:.1} passes ({:.1} single, {:.1} window, {:.1} split; \
             for {:.1} draw, {:.1} indexed, {:.1} dispatch, {:.1} mesh commands) over {:.1} records, {:.1} commands \
             needed none; processor, all recording threads: recording the passes {:.2} ms ({:.2} us single, {:.2} \
             window, {:.2} split), preparing the command after a pass {:.2} ms ({:.2} us each), preparing one without \
             a pass {:.2} ms ({:.2} us each), returning the buffer {:.2} ms",
            single.0, window.0, split.0,
            kinds[0], kinds[1], kinds[2], kinds[3],
            records as f64 / frames, within.0,
            single.1 + window.1 + split.1, single.2, window.2,
            split.2, after.1, after.2, within.1, within.2,
            done.1
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_cost_of_the_passes_is_counted_by_part_and_kind_then_cleared() {
        use cost::Part;
        let now = std::time::Instant::now();
        cost::take();
        cost::pass(Part::Single, 10, 1, now);
        cost::pass(Part::Window, 0, 16, now);
        cost::pass(Part::Split, 2, 1, now);
        cost::took(Part::AfterPass, now);
        cost::took(Part::InWindow, now);
        cost::took(Part::Done, now);
        cost::took(Part::Done, now);
        let (parts, records, kinds) = cost::take();
        assert_eq!(parts.map(|p| p.0), [1, 1, 1, 1, 1, 2]);
        assert_eq!(records, 18);
        assert_eq!(kinds, [1, 0, 1, 1]);
        assert_eq!(cost::take().1, 0);
        assert_eq!([0, 1, 2, 10].map(cost::kind_name), ["draw", "indexed", "dispatch", "mesh"]);
    }
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pages whose bytes are not
    /// what was hashed last and leaves the hashes: a page a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            assert!(std::env::var_os("G6_D3D12_WARP").is_none(), "requested WARP device could not be created");
            eprintln!("WARP D3D12 adapter unavailable; skipped card compare test");
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PAGE: usize = 4096;
        const UNWRITTEN: u32 = 0xdead_beef;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(crate::shadow::card_hash_bytes(size as u64));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // Read the whole output so a misplaced write cannot pass the test.
        let run_span = |warp: &mut WarpHarness, mode: u32, first: u32, skew: u32, pieces: u32, output: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(UNWRITTEN);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [first, skew, size as u32, pieces, output * 4, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(pieces, 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let run = |warp: &mut WarpHarness, mode| run_span(warp, mode, 0, 0, PIECES, 0);
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [UNWRITTEN; 3]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);

        // The allocation starts 16 bytes before a piece boundary. Run a
        // single middle group with a nonzero base and output offset.
        let middle = |warp: &mut WarpHarness, mode| run_span(warp, mode, 1, PIECE as u32 - 16, 1, 1);
        assert_eq!(middle(&mut warp, 0), [UNWRITTEN; 3]);
        let write = |at: usize, value| {
            // SAFETY: tests use offsets within the live buffer, between completed passes.
            assert!(at < size);
            unsafe { now_ptr.add(at).write_volatile(value) };
        };
        // Probe every page and all four 16-byte loads in its final
        // 64-byte block. The first page of piece 1 starts at buffer +16.
        for lane in 0..64 {
            let page = lane / 4;
            let byte = 16 + (page + 1) * PAGE - 64 + (lane % 4) * 16 + 15;
            write(byte, content[byte] ^ 0x80);
            assert_eq!(middle(&mut warp, 1), [UNWRITTEN, 1 << page, UNWRITTEN], "lane {lane}");
            write(byte, content[byte]);
        }
        for byte in [15, PIECE + 16] {
            write(byte, content[byte] ^ 1);
            assert_eq!(middle(&mut warp, 1), [UNWRITTEN, 0, UNWRITTEN]);
            write(byte, content[byte]);
        }
        // Compare-and-update must OR separate pages and remember them
        // when multiple pages change back in the following submission.
        for mask in [0x8181u32, 0xffff] {
            for page in 0..16 {
                if mask & (1 << page) != 0 {
                    let byte = 16 + page * PAGE;
                    write(byte, content[byte] ^ 1);
                }
            }
            assert_eq!(middle(&mut warp, 2), [UNWRITTEN, mask, UNWRITTEN]);
            assert_eq!(middle(&mut warp, 2), [UNWRITTEN, 0, UNWRITTEN]);
            for page in 0..16 {
                let byte = 16 + page * PAGE;
                write(byte, content[byte]);
            }
            assert_eq!(middle(&mut warp, 2), [UNWRITTEN, mask, UNWRITTEN]);
            assert_eq!(middle(&mut warp, 2), [UNWRITTEN, 0, UNWRITTEN]);
        }
        // A clipped first piece only holds the last 16 bytes of guest
        // page 15. Its flag retains the guest page index.
        let first = |warp: &mut WarpHarness, mode| run_span(warp, mode, 0, PIECE as u32 - 16, 1, 0);
        assert_eq!(first(&mut warp, 0), [UNWRITTEN; 3]);
        write(0, content[0] ^ 1);
        assert_eq!(first(&mut warp, 2), [1 << 15, UNWRITTEN, UNWRITTEN]);
        write(0, content[0]);
        assert_eq!(first(&mut warp, 2), [1 << 15, UNWRITTEN, UNWRITTEN]);
        // The final piece is short at the other end of the allocation.
        let tail = |warp: &mut WarpHarness, mode| run_span(warp, mode, 3, PIECE as u32 - 16, 1, 0);
        assert_eq!(tail(&mut warp, 0), [UNWRITTEN; 3]);
        for byte in [2 * PIECE + 16, size - 1] {
            write(byte, content[byte] ^ 1);
        }
        assert_eq!(tail(&mut warp, 2), [0x8001, UNWRITTEN, UNWRITTEN]);
        assert_eq!(tail(&mut warp, 2), [0, UNWRITTEN, UNWRITTEN]);
    }

    #[test]
    fn warp_card_fetch_preserves_gaps_and_resource_boundaries() {
        let Some(mut warp) = WarpHarness::new() else {
            assert!(std::env::var_os("G6_D3D12_WARP").is_none(), "requested WARP device could not be created");
            return;
        };
        let (a, src_a, _) = warp.shared_buffer(256);
        let (b, src_b, _) = warp.shared_buffer(256);
        let (out, bytes, _) = warp.shared_buffer(256);
        let first: Vec<u8> = (0..=255).collect();
        let second: Vec<u8> = (0..=255).rev().collect();
        let mut expected = [0xcdu8; 256];
        // SAFETY: mapped buffers of exactly 256 bytes; no work is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(first.as_ptr(), src_a, 256);
            std::ptr::copy_nonoverlapping(second.as_ptr(), src_b, 256);
            bytes.write_bytes(0xcd, 256);
        }
        let slices = [
            (&a, 1, 10, 4), (&a, 10, 18, 13), // adjacent in both
            (&a, 18, 29, 24),                  // gap only in destination
            (&a, 32, 39, 35),                  // gap only in source
            (&b, 39, 47, 42),                  // different source
            (&a, 4, 9, 50),                    // reused range of the first source
        ];
        for (resource, lo, hi, at) in slices {
            let source = if resource == &a { &first } else { &second };
            expected[at as usize..at as usize + hi - lo].copy_from_slice(&source[lo..hi]);
        }
        crate::host::copy_fetch_runs(&warp.list, &out, slices.into_iter());
        warp.submit_and_wait();
        // SAFETY: the destination is still alive and every copy has completed.
        assert_eq!(unsafe { std::slice::from_raw_parts(bytes, 256) }, expected);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes,
        // then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// One dispatch in parts: groups past what the host runs in one
    /// dimension go into the second record; what nothing wrote stays
    /// clamped, or empty.
    #[test]
    fn warp_large_dispatch_is_made_in_parts() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let cases: [([u32; 3], [u32; 6]); 6] = [
            ([115_200, 1, 1], [65_535, 1, 1, 49_665, 1, 1]),
            ([65_535, 1, 1], [65_535, 1, 1, 0, 1, 1]),
            ([65_536, 1, 1], [65_535, 1, 1, 1, 1, 1]),
            ([5, 3, 2], [5, 3, 2, 0, 3, 2]),
            ([0, 1, 1], [0, 1, 1, 0, 1, 1]),
            // Past both parts: as before, what one dispatch runs.
            ([200_000, 1, 1], [65_535, 1, 1, 0, 1, 1]),
        ];
        for (asked, parts) in cases {
            let (src_buf, src_ptr, _) = warp.shared_buffer(16);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(64);
            let dst_buf = warp.default_uav(64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, 64);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (2, 12, 1),
                IN_PARTS,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list and live buffers.
            unsafe { warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 24) };
            warp.submit_and_wait();
            // SAFETY: the mapping holds 24 bytes the copy wrote.
            let made: [u32; 6] = unsafe { readback_ptr.cast::<[u32; 6]>().read_volatile() };
            assert_eq!(made, parts, "{asked:?}");
        }
    }

    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                if ahead { AHEAD } else { 0 },
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list; the scratch buffer is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// The records of a group lie in state `before`: copied to `readback`
    /// and left as they were.
    fn copy_out(l: &ID3D12GraphicsCommandList, from: &ID3D12Resource, before: D3D12_RESOURCE_STATES, readback: &ID3D12Resource, at: u64, bytes: u64) {
        // SAFETY: an open list and live buffers of those sizes.
        unsafe {
            l.ResourceBarrier(&[transition(from, before, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyBufferRegion(readback, at, from, 0, bytes);
            l.ResourceBarrier(&[transition(from, D3D12_RESOURCE_STATE_COPY_SOURCE, before)]);
        }
    }

    #[test]
    fn warp_grouped_clamp_leaves_every_record_as_the_clamp_of_one_does() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let grouped = make_from(&warp.device, batch::SOURCE).expect("the grouped clamp compiles on WARP");
        // Draws, indexed draws and mesh dispatches, with garbage among them;
        // their costs stay under the budget, so that order does not matter.
        let records: [(u32, [u32; 5]); 12] = [
            (0, [3, 10, 7, 2, 0xAAAA]),
            (0, [u32::MAX, 1, 1, 9, 0xBBBB]),
            (0, [100, u32::MAX, 5, 6, 0]),
            (0, [0, 5, 4, 3, 0]),
            (1, [36, 2, 100, (-4i32) as u32, 7]),
            (1, [u32::MAX, 2, 11, 12, 13]),
            (1, [6, 0, 1, 2, 3]),
            (10, [1, 1, 1, 77, 78]),
            (10, [u32::MAX, 1, 1, 0, 0]),
            (10, [200, 5, 9, 0, 0]),
            (10, [0, 1, 1, 0, 0]),
            (10, [65_536, 0, 7, 0, 0]),
        ];
        const STRIDE: u64 = 32;
        let bytes = STRIDE * records.len() as u64;
        // Half of them from a second buffer.
        let sources = [warp.shared_buffer(bytes), warp.shared_buffer(bytes)];
        let source_of = |i: usize| i % 2;
        for (i, (_, words)) in records.iter().enumerate() {
            // SAFETY: inside the mapped buffer.
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), sources[source_of(i)].1.add(i * STRIDE as usize).cast::<u32>(), 5) };
        }
        let (_one_budget, one_budget_ptr, one_budget) = warp.shared_buffer(256);
        let (_all_budget, all_budget_ptr, all_budget) = warp.shared_buffer(256);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        let (one_back, one_ptr, _) = warp.shared_buffer(bytes);
        let (all_back, all_ptr, _) = warp.shared_buffer(batch::BUFFER);
        // SAFETY: mapped buffers of at least 256 bytes.
        unsafe {
            std::ptr::write_bytes(one_budget_ptr, 0, 256);
            std::ptr::write_bytes(all_budget_ptr, 0, 256);
        }
        let single = warp.default_uav(STRIDE);
        let group = warp.default_uav(batch::BUFFER);
        for (i, (kind, _)) in records.iter().enumerate() {
            let at = i as u64 * STRIDE;
            assert!(record_clamp(&warp.clamp, &warp.list, &single, one_budget, (&sources[source_of(i)].0, at), None, (*kind, STRIDE as u32, 1)));
            copy_out(&warp.list, &single, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, &one_back, at, STRIDE);
            done_on(&warp.list, &single);
        }
        // SAFETY: the table's mapped upload-sized buffer, read when the list
        // runs, after the jobs are added.
        let mut table = unsafe { batch::Table::new(table_ptr, [11, 12, 0, 0]) };
        batch::record(&grouped, &warp.list, (&group, D3D12_RESOURCE_STATE_COMMON), all_budget, [sources[0].2, sources[1].2, sources[0].2, sources[0].2], table_gpu);
        let places: Vec<u64> = records
            .iter()
            .enumerate()
            .map(|(i, (kind, _))| table.add(11 + source_of(i) as u64, i as u64 * STRIDE, *kind).expect("a job"))
            .collect();
        copy_out(&warp.list, &group, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, &all_back, 0, batch::BUFFER);
        warp.submit_and_wait();
        let word = |ptr: *mut u8, at: u64| {
            // SAFETY: inside a mapped read-back buffer the finished list wrote.
            unsafe { ptr.add(at as usize).cast::<u32>().read_volatile() }
        };
        for (i, (kind, _)) in records.iter().enumerate() {
            let words = u64::from(min_stride(*kind).unwrap()) / 4;
            let one: Vec<u32> = (0..words).map(|w| word(one_ptr, i as u64 * STRIDE + w * 4)).collect();
            let all: Vec<u32> = (0..words).map(|w| word(all_ptr, places[i] + w * 4)).collect();
            assert_eq!(all, one, "record {i} of kind {kind}");
        }
        // The same work charged, and the same noted about mesh records.
        for at in [0, 4, 8] {
            assert_eq!(word(all_budget_ptr, at), word(one_budget_ptr, at), "budget word at {at}");
        }
        assert!(word(all_budget_ptr, 0) > 0 && word(all_budget_ptr, 0) < 1 << 27);
    }

    #[test]
    fn warp_grouped_clamp_stops_at_the_budget() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let grouped = make_from(&warp.device, batch::SOURCE).expect("the grouped clamp compiles on WARP");
        // 128 indexed draws that each cost half the budget.
        const RECORDS: u64 = 128;
        const STRIDE: u64 = 20;
        let (source, source_ptr, source_gpu) = warp.shared_buffer(RECORDS * STRIDE);
        for i in 0..RECORDS {
            let words: [u32; 5] = [u32::MAX, u32::MAX, 1000 + i as u32, 2000 + i as u32, 3000 + i as u32];
            // SAFETY: inside the mapped buffer.
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), source_ptr.add((i * STRIDE) as usize).cast::<u32>(), 5) };
        }
        let (_budget, budget_ptr, budget) = warp.shared_buffer(256);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        let (back, back_ptr, _) = warp.shared_buffer(batch::BUFFER);
        // SAFETY: a mapped buffer of at least 256 bytes.
        unsafe { std::ptr::write_bytes(budget_ptr, 0, 256) };
        let group = warp.default_uav(batch::BUFFER);
        // SAFETY: as in the test above.
        let mut table = unsafe { batch::Table::new(table_ptr, [5, 0, 0, 0]) };
        batch::record(&grouped, &warp.list, (&group, D3D12_RESOURCE_STATE_COMMON), budget, [source_gpu; 4], table_gpu);
        let places: Vec<u64> = (0..RECORDS).map(|i| table.add(5, i * STRIDE, 1).expect("a job")).collect();
        copy_out(&warp.list, &group, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, &back, 0, batch::BUFFER);
        warp.submit_and_wait();
        drop(source);
        // SAFETY: the finished list's read-back and budget buffers.
        let (spent, out) = unsafe {
            (budget_ptr.cast::<u32>().read_volatile(), std::slice::from_raw_parts(back_ptr.cast::<u32>(), batch::BUFFER as usize / 4).to_vec())
        };
        assert_eq!(spent, 1 << 27, "the budget saturates without wrapping");
        let mut kept = 0;
        for (i, place) in places.iter().enumerate() {
            let record = &out[*place as usize / 4..][..5];
            if record[..2] == [1 << 24, 4] {
                kept += 1;
            } else {
                assert_eq!(record[..2], [0, 0], "an excess draw is empty");
            }
            assert_eq!(record[2..], [1000 + i as u32, 2000 + i as u32, 3000 + i as u32]);
        }
        assert_eq!(kept, 2);
    }

    #[test]
    fn a_group_takes_only_what_it_can_bound_and_no_more_than_it_holds() {
        let mut memory = vec![0u32; batch::TABLE as usize / 4];
        // SAFETY: that many bytes, held by `memory` while the table is used.
        let mut table = unsafe { batch::Table::new(memory.as_mut_ptr().cast(), [3, 4, 0, 0]) };
        assert_eq!(table.add(9, 0, 1), None, "a buffer the pass does not read");
        assert_eq!(table.add(0, 0, 1), None, "no buffer");
        assert_eq!(table.add(3, 2, 1), None, "unaligned");
        assert_eq!(table.add(3, 0, 2), None, "a dispatch");
        assert_eq!(table.add(3, u64::from(u32::MAX) - 3, 10), None, "past what the pass addresses");
        assert_eq!(table.add(4, 40, 10), Some(0));
        assert_eq!(table.add(3, 80, 0), Some(u64::from(batch::SLOT)));
        for i in 2..batch::JOBS {
            assert_eq!(table.add(3, u64::from(i) * 20, 1), Some(u64::from(i * batch::SLOT)));
        }
        assert_eq!(table.add(3, 0, 1), None, "full");
        assert_eq!(memory[0], batch::JOBS);
        // Kind, source, offset, place.
        assert_eq!(memory[4..12], [10, 1, 40, 0, 0, 0, 80, batch::SLOT]);
    }

    /// `open_batch` on a host list: two groups in one list, then the list
    /// again after its reset, each leaving the records where the draws read.
    #[test]
    fn a_host_list_records_groups_again_after_its_reset() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().expect("a host list");
        let (source, source_ptr) = host.shared_buffer(256).expect("a source");
        let (back, back_ptr) = host.shared_buffer(2 * batch::BUFFER).expect("a read-back buffer");
        let words: [u32; 5] = [30, 2, 7, 8, 9];
        // SAFETY: inside the mapped buffer.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), source_ptr.cast::<u32>(), 5) };
        for round in 0..2 {
            let mut places = Vec::new();
            for group in 0..2u64 {
                let (mut table, records) = open_batch(&list, (21, &source), &[]).expect("a grouped pass");
                places.push(table.add(21, 0, 1).expect("a job"));
                copy_out(&list.list, &records, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, &back, group * batch::BUFFER, batch::BUFFER);
            }
            list.close();
            host.execute(&[&list]);
            for (group, place) in places.iter().enumerate() {
                // SAFETY: the finished list's read-back buffer.
                let out = unsafe {
                    std::slice::from_raw_parts(back_ptr.add(group * batch::BUFFER as usize + *place as usize).cast::<u32>(), 5).to_vec()
                };
                assert_eq!(out, words, "round {round} group {group}");
            }
            list.reset();
        }
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        done(&list, &b1);
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations.
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
    }
}

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p g6-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
        /// The pass of ours, its words in the PC's memory.
        ComputeInPc,
        /// The clamp of one mesh record, with its budget in the card's
        /// memory or in the PC's.
        Clamp { pc: bool },
    }

    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        words_pc: ID3D12Resource,
        clamp: super::Clamp,
        /// The records the clamp reads, and where it writes them.
        args: ID3D12Resource,
        scratch: ID3D12Resource,
        /// In the card's memory, in the PC's.
        budgets: [ID3D12Resource; 2],
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        _depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    /// A buffer shaders write in the PC's memory, as `Host::shared_buffer`
    /// makes them.
    fn pc_buffer(device: &ID3D12Device, size: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let words_pc = pc_buffer(&device, 64 << 10);
                let clamp = super::make_for(&device).ok()?;
                let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                let made = |flags| buffer(&device, 64 << 10, D3D12_HEAP_TYPE_DEFAULT, flags, D3D12_RESOURCE_STATE_COMMON);
                let (args, scratch) = (made(D3D12_RESOURCE_FLAG_NONE), made(uav));
                let budgets = [made(uav), pc_buffer(&device, 64 << 10)];
                // One group of one: what the clamp reads.
                let one = buffer(&device, 64 << 10, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                let mut mapped = std::ptr::null_mut();
                one.Map(0, None, Some(&mut mapped)).ok()?;
                mapped.cast::<[u32; 3]>().write([1, 1, 1]);
                one.Unmap(0, None);
                list.CopyBufferRegion(&args, 0, &one, 0, 12);
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    device: device.clone(),
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    words_pc,
                    clamp,
                    args,
                    scratch,
                    budgets,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image, one],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    _depth_written: depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// The milliseconds between the two timestamps the last list resolved.
        fn stamped_ms(&self) -> f64 {
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer a finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            ticks as f64 * 1e3 / frequency as f64
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                            Between::ComputeInPc => {
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words_pc.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                            Between::Clamp { pc } => {
                                let budget = self.budgets[usize::from(pc)].GetGPUVirtualAddress();
                                assert!(super::record_clamp(&self.clamp, l, &self.scratch, budget, (&self.args, 0), None, (10, 12, 1)));
                                super::done_on(l, &self.scratch);
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    /// What the compare of a title buffer's copy on the card costs: the
    /// 174 MB buffer of the title's world, hashed whole, as often as a frame
    /// of the prison scene asks (about 13,900 pieces of 64 KiB).
    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn card_compare_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter; skipped");
            return;
        };
        let code = compile(crate::host::CARD_COMPARE, s!("main"), s!("cs_5_1")).expect("the card compare");
        // SAFETY: the compiled program carries its root signature; the
        // pipeline description borrows it and the code for the call.
        let compare = unsafe {
            let root: ID3D12RootSignature = rig.device.CreateRootSignature(0, &code[..]).expect("root signature");
            let pso: ID3D12PipelineState = rig
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                    ..Default::default()
                })
                .expect("pipeline");
            super::Clamp { root, pso }
        };
        let size = 174u64 << 20;
        let pieces = (size >> 16) as u32;
        let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let made = |bytes| buffer(&rig.device, bytes, D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_COMMON);
        let (now, hashes, differs) = (made(size), made(crate::shadow::card_hash_bytes(size)), made(64 << 10));
        for times in [1u32, 5] {
            let mut took = Vec::new();
            for _ in 0..6 {
                let l = &rig.list;
                // SAFETY: an open list, the compare's own root signature,
                // live buffers (promoted from the common state).
                unsafe {
                    l.EndQuery(&rig.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    l.SetComputeRootSignature(&compare.root);
                    l.SetPipelineState(&compare.pso);
                    l.SetComputeRootShaderResourceView(1, now.GetGPUVirtualAddress());
                    l.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
                    l.SetComputeRootUnorderedAccessView(3, differs.GetGPUVirtualAddress());
                    for _ in 0..times {
                        let constants = [0u32, 0, size as u32, pieces, 0, 2];
                        l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        l.Dispatch(pieces, 1, 1);
                        l.ResourceBarrier(&[super::uav_barrier()]);
                    }
                    l.EndQuery(&rig.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    l.ResolveQueryData(&rig.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &rig.back, 0);
                }
                rig.finish();
                took.push(rig.stamped_ms());
            }
            took.sort_by(f64::total_cmp);
            eprintln!(
                "card compare of {pieces} pieces {times} time(s) ({} MB): {:.2} ms on the card (median of {})",
                u64::from(times) * (size >> 20),
                took[took.len() / 2],
                took.len()
            );
        }
    }

    /// What the grouped clamp costs a draw, with its budget in the card's
    /// memory or in the PC's, against the clamp of one record before every
    /// draw; and whether a clamp between draws that have work of their own
    /// costs more than the clamp alone (the card no longer running them
    /// over each other).
    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn grouped_clamp_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        let grouped = super::make_from(&rig.device, super::batch::SOURCE).expect("the grouped clamp");
        const PER: u32 = 2048;
        let table_bytes = super::batch::TABLE.next_multiple_of(256);
        let tables = buffer(&rig.device, table_bytes * u64::from(PER), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: an upload buffer of ours, mapped for the test's life.
        unsafe { tables.Map(0, None, Some(&mut mapped)).expect("map") };
        let out = buffer(&rig.device, super::batch::BUFFER, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        eprintln!("grouped clamp on {}: {PER} draws a batch, least of three", rig.card);
        // A pixel area and instances that give a draw some microseconds of
        // work of its own, or none.
        for (work, side, instances) in [("a bare triangle", 0u32, 1u32), ("200 triangles on 256x256", 256, 200)] {
            let run = |rig: &mut Rig, how: u32, group: u32, pc: bool| -> f64 {
                let mut best = f64::MAX;
                for _ in 0..3 {
                    rig.begin();
                    let budget = rig.budgets[usize::from(pc)].clone();
                    let mut before = D3D12_RESOURCE_STATE_COMMON;
                    // SAFETY: an open list with everything set; live buffers;
                    // the tables lie in the mapped upload buffer, one a pass.
                    unsafe {
                        let l = rig.list.clone();
                        if side > 0 {
                            l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: side as f32, Height: side as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
                        }
                        l.EndQuery(&rig.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                        let mut passes = 0u64;
                        for draw in 0..PER {
                            if how == 1 {
                                assert!(super::record_clamp(&rig.clamp, &l, &rig.scratch, budget.GetGPUVirtualAddress(), (&rig.args, 0), None, (10, 12, 1)));
                                super::done_on(&l, &rig.scratch);
                                l.SetPipelineState(&rig.pipelines[0]);
                            } else if how == 2 && draw % group == 0 {
                                let at = passes * table_bytes;
                                let mut table = super::batch::Table::new(mapped.cast::<u8>().add(at as usize), [1, 0, 0, 0]);
                                for _ in 0..group {
                                    table.add(1, 0, 10).expect("a job");
                                }
                                let source = rig.args.GetGPUVirtualAddress();
                                super::batch::record(&grouped, &l, (&out, before), budget.GetGPUVirtualAddress(), [source; super::batch::SOURCES], tables.GetGPUVirtualAddress() + at);
                                before = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
                                passes += 1;
                                l.SetPipelineState(&rig.pipelines[0]);
                            }
                            l.DrawInstanced(3, instances, 0, 0);
                        }
                        l.EndQuery(&rig.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&rig.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &rig.back, 0);
                    }
                    rig.finish();
                    best = best.min(rig.stamped_ms() * 1e3 / f64::from(PER));
                }
                best
            };
            let bare = run(&mut rig, 0, 1, false);
            eprintln!("== {work}: no clamp {bare:.2} us a draw");
            for pc in [true, false] {
                let place = if pc { "PC's memory" } else { "card's memory" };
                eprintln!("   one clamp a draw, budget in the {place}: {:.2} us a draw", run(&mut rig, 1, 1, pc));
                for group in [4u32, 16, 64, 256] {
                    eprintln!("   one pass for {group} draws, budget in the {place}: {:.2} us a draw", run(&mut rig, 2, group, pc));
                }
            }
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
            ("(j) the pass of (h), its words in the PC's memory", Between::ComputeInPc, 1),
            ("(k) the clamp of one mesh record before every draw, budget in the card's memory", Between::Clamp { pc: false }, 1),
            ("(l) the same clamp, budget in the PC's memory", Between::Clamp { pc: true }, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
