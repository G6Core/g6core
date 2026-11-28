//! One pass that bounds the records of many indirect draws (`crate::draw`;
//! `G6_INDIRECT_BATCH=0` turns the groups off).
//!
//! The idea is another project's (`CREDITS.md`), written here for G6's clamp:
//! each job is bounded by the rules [`super::SOURCE`] bounds one record of one
//! command by. Only draws are taken (kinds 0, 1 and 10).
//!
//! The draws after the one a pass is recorded for are not known yet, so the
//! pass reads what to bound from a table of jobs in upload memory, which the
//! processor goes on filling while the list is recorded: the pass is as wide
//! as the table and its threads past the count do nothing. Each draw's record
//! gets a slot of its own in the pass's buffer.

use windows::Win32::Graphics::Direct3D12::*;

use super::{Clamp, min_stride, uav_barrier};
use crate::host::transition;

/// The jobs of one pass.
pub(crate) const JOBS: u32 = 256;
/// The bytes of its table in upload memory: the count, then the jobs.
pub(crate) const TABLE: u64 = 16 + 16 * JOBS as u64;
/// The bytes of a bounded record's slot.
pub(crate) const SLOT: u32 = 32;
/// The title buffers a pass can read.
pub(crate) const SOURCES: usize = 4;
/// The bytes of the buffer the bounded records of one pass lie in.
pub(crate) const BUFFER: u64 = JOBS as u64 * SLOT as u64;

pub(super) const SOURCE: &str = r#"
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1)"
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
ByteAddressBuffer src2 : register(t2);
ByteAddressBuffer src3 : register(t3);
// The count, 12 bytes of nothing, then per job: kind, source, the record's
// offset in the source, its offset in the destination.
ByteAddressBuffer jobs : register(t4);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
uint load(uint source, uint at) {
    if (source == 0) return src0.Load(at);
    if (source == 1) return src1.Load(at);
    if (source == 2) return src2.Load(at);
    return src3.Load(at);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= jobs.Load(0)) return;
    uint4 job = jobs.Load4(16 + id.x * 16);
    uint kind = job.x;
    uint source = job.y;
    uint src_at = job.z;
    uint dst_at = job.w;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(load(source, src_at), 1u << 24);
        uint instances = min(load(source, src_at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = uint3(load(source, src_at), load(source, src_at + 4), load(source, src_at + 8));
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
        // A draw's record is four words, an indexed draw's five.
        uint words = kind == 0 ? 4u : 5u;
        for (uint w = 2; w < words; w++) {
            dst_args.Store(dst_at + w * 4, load(source, src_at + w * 4));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
    }
}
"#;

/// The jobs of a pass as the processor fills them ([`SOURCE`]).
pub(crate) struct Table {
    /// The table in upload memory ([`TABLE`] bytes).
    at: usize,
    /// The title buffer copies the pass can read, by their number
    /// (`crate::shadow::Shadow::id`; 0: none).
    sources: [u64; SOURCES],
    jobs: u32,
}

impl Table {
    /// A table at `at`, with no job yet.
    ///
    /// # Safety
    /// `at` is [`TABLE`] bytes of mapped upload memory, 4-byte aligned, that
    /// live until every submission of the list that reads them has run, and
    /// that list is not run while jobs are added.
    pub(crate) unsafe fn new(at: *mut u8, sources: [u64; SOURCES]) -> Table {
        // SAFETY: the caller's.
        unsafe { at.cast::<u32>().write_volatile(0) };
        Table { at: at as usize, sources, jobs: 0 }
    }

    /// Adds the record of `kind` at `offset` of the copy `id`: where it lies
    /// in the pass's buffer once the pass has run. `None` if the pass does
    /// not read that copy, the record is not one it takes, or it is full.
    pub(crate) fn add(&mut self, id: u64, offset: u64, kind: u32) -> Option<u64> {
        if id == 0 || !matches!(kind, 0 | 1 | 10) || self.jobs == JOBS {
            return None;
        }
        let source = self.sources.iter().position(|&s| s == id)? as u32;
        let offset = u32::try_from(offset).ok().filter(|o| o.is_multiple_of(4))?;
        offset.checked_add(min_stride(kind)?)?;
        let place = self.jobs * SLOT;
        let table = self.at as *mut u32;
        // SAFETY: inside the table ([`Table::new`]): fewer than [`JOBS`]
        // jobs before this one. The count last, so that it never names a job
        // not yet written.
        unsafe {
            let to = table.add(4 + 4 * self.jobs as usize);
            to.write_volatile(kind);
            to.add(1).write_volatile(source);
            to.add(2).write_volatile(offset);
            to.add(3).write_volatile(place);
            self.jobs += 1;
            table.write_volatile(self.jobs);
        }
        Some(u64::from(place))
    }
}

/// Records the pass over the table at `jobs` (a host address) into `dst`,
/// which is in state `before` and left in `INDIRECT_ARGUMENT`: after every
/// earlier shader write (the title's argument passes), before the draws that
/// read the bounded records. The pass's own writes of the budget are waited
/// for by the next pass's first barrier, as every pass of ours waits.
pub(super) fn record(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    sources: [u64; SOURCES],
    jobs: u64,
) {
    let first = [uav_barrier(), transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)];
    let after = [transition(dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)];
    // SAFETY: an open list; our root signature and pipeline; live buffers.
    // The sources and the table are read in the common state (promoted).
    unsafe {
        l.ResourceBarrier(&first);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for (i, source) in sources.into_iter().enumerate() {
            l.SetComputeRootShaderResourceView(i as u32, source);
        }
        l.SetComputeRootShaderResourceView(SOURCES as u32, jobs);
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 1, dst.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 2, budget);
        l.Dispatch(JOBS.div_ceil(64), 1, 1);
        l.ResourceBarrier(&after);
    }
}
