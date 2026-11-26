//! Packed 16-bit operations (`v_pk_*`), the byte dot products, and what
//! else the upscaler's programs need.
use super::super::bitcode::{Attr, Bin, V, pred};
use super::{Lower, Why};
use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Inst, Reg, Value};
use crate::resource::{Role, Site, Source};

impl<'a> Lower<'a> {
    /// Whether the mask operation that sets the SCC value `scc` gives no
    /// lane outside EXEC a bit.
    pub(super) fn within_exec(&self, scc: Value) -> bool {
        let Some(&b) = self.def_block.get(&scc) else {
            return false;
        };
        let by = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.write(Reg::Scc) == Some(scc));
        by.is_some_and(|i| {
            i.write(Reg::S(EXEC)).is_some()
                || matches!(&i.fields, Fields::Sop { dst: Some(Operand::Scalar { code, .. }), .. }
                    if i.write(Reg::S(*code)).is_some_and(|lo| self.x.st.active.contains(&lo)))
        })
    }

    /// Whether the image `src` names is one of 32-bit integers (one channel):
    /// the program has an integer atomic on the same table entry, which
    /// takes no other view.
    pub(super) fn is_counter(&self, src: &Source) -> bool {
        let entry = |s: &Source| match s {
            Source::Loaded { load, first } => self
                .scalar
                .get(load)
                .filter(|a| a.dynamic.is_none())
                .map(|a| (a.param, a.offset + 4 * u32::from(*first))),
            _ => None,
        };
        let Some(this) = entry(src) else {
            return false;
        };
        self.x.res.descriptors.iter().any(|d| {
            let i = &self.x.p.blocks[d.site.0].insts[d.site.1];
            d.role == Role::Resource
                && matches!(i.fields, Fields::Image { dmask: 1, .. })
                && matches!(
                    i.name(),
                    "image_atomic_add"
                        | "image_atomic_sub"
                        | "image_atomic_and"
                        | "image_atomic_or"
                        | "image_atomic_xor"
                        | "image_atomic_smin"
                        | "image_atomic_smax"
                        | "image_atomic_umin"
                        | "image_atomic_umax"
                )
                && entry(&d.source) == Some(this)
        })
    }

    /// A one-channel store of `value` into a counter image ([`Self::is_counter`])
    /// of dimension `kind`, at `coords`.
    pub(super) fn store_counter(
        &mut self,
        value: Value,
        d16: bool,
        (site, src, kind): (Site, &Source, u32),
        coords: Vec<V>,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let x = self.i32_of(value)?;
        // A 16-bit integer, extended by the view's sign: the same number
        // either way only if it is known to be small.
        if d16 && !self.s.m.int_k(x).is_some_and(|(_, k)| k < 0x8000) {
            return Err("a 16-bit image store of one channel");
        }
        let h = self.handle(site, src, false, (kind | 0x1000, 0x105))?;
        let at = self.begin_guarded(guard);
        // All four channels are written (the validator wants them all): the
        // view has the first alone.
        let mut args = vec![h];
        args.extend(coords);
        args.extend([x; 4]);
        args.push(self.s.m.i8(0xF));
        let void = self.s.t.void;
        self.s.op(
            self.cur,
            "textureStore.i32",
            Attr::NoUnwind,
            void,
            67,
            &args,
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Half `high` of source `k` of `v_pk_fmac_f16`, in the low 16 bits: a
    /// VOP2 operation (no op_sel) whose third source is its destination.
    fn fmac_half(&mut self, inst: &Inst, k: usize, high: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = if k == 2 { v.dst } else { v.src.get(k).copied() };
        let op = op.ok_or("a missing operand")?;
        if matches!(op, Operand::Const(_) | Operand::Literal(_)) {
            // shortcut: no constant operand seen; its high half is not known.
            return Err("a constant operand of v_pk_fmac_f16");
        }
        let x = self.operand(inst, &op)?;
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// `v_dot4c_i32_i8` and `v_dot4_i32_i8`: an accumulator plus the dot
    /// product of four signed bytes, wrapping (DXIL's `Dot4AddI8Packed`).
    pub(super) fn dot4(&mut self, inst: &Inst, name: &str) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let (acc, a, b) = if name == "v_dot4c_i32_i8" {
            // VOP2: the accumulator is the destination.
            let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
            let Some(Operand::Vector { reg, .. }) = v.dst else {
                return Err("an accumulator");
            };
            let acc = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
            (acc, a, b)
        } else {
            // VOP3P: op_sel picks the half each multiplicand's low half
            // reads, op_sel_hi its high half's; the accumulator is read
            // whole (as AMD's compilers fold constants into it).
            if v.clamp || v.neg_hi != 0 {
                return Err("a saturating or negated dot product");
            }
            let mut xs = Vec::new();
            for k in 0..2 {
                let x = self.src(inst, k)?;
                let (lo, hi) = (v.opsel >> k & 1 != 0, v.opsel_hi >> k & 1 != 0);
                if (lo, hi) == (false, true) {
                    xs.push(x);
                    continue;
                }
                let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                let top = self.s.m.bin(c, Bin::LShr, x, k16);
                let bottom = self.s.m.bin(c, Bin::And, x, low);
                let upper = self
                    .s
                    .m
                    .bin(c, Bin::Shl, if hi { top } else { bottom }, k16);
                xs.push(
                    self.s
                        .m
                        .bin(c, Bin::Or, if lo { top } else { bottom }, upper),
                );
            }
            let acc = self.src(inst, 2)?;
            (acc, xs[0], xs[1])
        };
        let i32_ty = self.s.t.i32;
        Ok(self.s.op(
            c,
            "dot4AddPacked.i32",
            Attr::ReadNone,
            i32_ty,
            163,
            &[acc, a, b],
        ))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    pub(super) fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "fmac_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = if op == "fmac_f16" {
                    self.fmac_half(inst, k, high)?
                } else {
                    self.packed_src(inst, k, high, float)?
                };
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }
}
