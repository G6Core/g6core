//! Fetching ahead, off unless `G6_FETCH_AHEAD` asks: the pieces of card
//! copies a submission is forecast to change ([`crate::forecast`]) are also
//! copied to PC memory right behind it.
//!
//! `verify`: nothing is merged from these early copies. When the submission
//! lands, the usual fetch runs and the two are compared. The log says how
//! many landings the early copies would have covered, and how often their
//! bytes were the fetched ones. A piece a later submission wrote before the
//! fetch ran differs rightly, so the share of identical pieces is a lower
//! bound on how sound the early copy is.
//!
//! `on`: a landing whose every changed piece lies in an early copy is merged
//! from those, with no fetch and no wait (`Host::relay_card_back`); any
//! other is fetched as before. What keeps that sound:
//!
//! - Every capture of card bytes, early copy or fetch, has a [`position`]
//!   in the order of the queue. A landing may bring bytes older than some
//!   already merged (an early copy made before a fetch that ran for an
//!   earlier landing); [`Versions`] keeps those out.
//! - A merge must know which title bytes had not reached the card when its
//!   bytes were captured, so the notes of those sends are kept as long as
//!   an early copy that needs them is not done with ([`bound`]).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Landings between two lines of the log.
const EVERY: u64 = 200;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Verify,
    On,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("G6_FETCH_AHEAD").as_deref() {
        Ok("verify") => Mode::Verify,
        Ok("on") => Mode::On,
        _ => Mode::Off,
    })
}

/// Whether early copies are made.
pub(crate) fn copying() -> bool {
    mode() != Mode::Off
}

/// Whether early copies are compared with the usual fetch.
pub(crate) fn verifying() -> bool {
    mode() == Mode::Verify
}

/// Whether landings are merged from early copies where they can be.
pub(crate) fn serving() -> bool {
    mode() == Mode::On
}

/// The place in the order of the queue of a capture of card bytes being sent
/// now, later ones higher. Called with the order of the queue held, between
/// the capture's send and the release.
pub(crate) fn position() -> u64 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    LAST.fetch_add(1, Ordering::Relaxed) + 1
}

/// Early copies not done with: how many by the number of the last send of
/// title bytes ahead of each.
static OUTSTANDING: Mutex<BTreeMap<u64, u32>> = Mutex::new(BTreeMap::new());

fn outstanding() -> MutexGuard<'static, BTreeMap<u64, u32>> {
    OUTSTANDING.lock().unwrap_or_else(|e| e.into_inner())
}

/// An early copy was sent behind send `after`. Called with the order of the
/// queue still held.
pub(crate) fn early_sent(after: u64) {
    *outstanding().entry(after).or_default() += 1;
}

/// That early copy will not be merged any more.
pub(crate) fn early_done(after: u64) {
    let mut held = outstanding();
    if let Some(count) = held.get_mut(&after) {
        *count -= 1;
        if *count == 0 {
            held.remove(&after);
        }
    }
}

/// `after`, or less if an early copy not done with was sent behind an
/// earlier send: notes of sends past this number must be kept.
pub(crate) fn bound(after: u64) -> u64 {
    outstanding().keys().next().map_or(after, |&oldest| oldest.min(after))
}

/// By copy, then by a range's start: its end and the position of the newest
/// capture merged there.
#[derive(Default)]
pub(crate) struct Versions(HashMap<usize, BTreeMap<usize, (usize, u64)>>);

impl Versions {
    /// Bytes of `[lo, hi)` of `copy` captured at `position` are to be merged:
    /// the parts of the range where nothing newer has been, which are noted
    /// as holding this capture now.
    pub(crate) fn claim(&mut self, copy: usize, lo: usize, hi: usize, position: u64) -> Vec<(usize, usize)> {
        if lo >= hi {
            return Vec::new();
        }
        let held = self.0.entry(copy).or_default();
        // The ranges that overlap: they lie apart and in order.
        let mut over: Vec<(usize, usize, u64)> = held
            .range(..hi)
            .rev()
            .take_while(|(_, (end, _))| *end > lo)
            .map(|(&start, &(end, at))| (start, end, at))
            .collect();
        over.reverse();
        let mut fresh: Vec<(usize, usize)> = Vec::new();
        let mut take = |from: usize, to: usize| match fresh.last_mut() {
            Some(last) if last.1 == from => last.1 = to,
            _ => fresh.push((from, to)),
        };
        let mut new: Vec<(usize, usize, u64)> = Vec::new();
        let mut reached = lo;
        for &(start, end, at) in &over {
            held.remove(&start);
            if start < lo {
                new.push((start, lo, at));
            }
            let (from, to) = (start.max(lo), end.min(hi));
            if reached < from {
                take(reached, from);
                new.push((reached, from, position));
            }
            if at <= position {
                take(from, to);
            }
            new.push((from, to, at.max(position)));
            reached = to;
            if end > hi {
                new.push((hi, end, at));
            }
        }
        if reached < hi {
            take(reached, hi);
            new.push((reached, hi, position));
        }
        // Back in, ranges that touch and hold the same capture as one.
        new.sort_unstable();
        let mut joined: Vec<(usize, usize, u64)> = Vec::new();
        for (start, end, at) in new {
            match joined.last_mut() {
                Some(last) if last.1 == start && last.2 == at => last.1 = end,
                _ => joined.push((start, end, at)),
            }
        }
        held.extend(joined.into_iter().map(|(start, end, at)| (start, (end, at))));
        fresh
    }
}

/// What has been merged where, for landings served `on`.
pub(crate) fn versions() -> MutexGuard<'static, Option<Versions>> {
    static VERSIONS: Mutex<Option<Versions>> = Mutex::new(None);
    VERSIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tallies a landing with changed pieces, served from early copies or not.
pub(crate) fn landed(early: bool) {
    static SEEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    SEEN[usize::from(early)].fetch_add(1, Ordering::Relaxed);
    let all = SEEN[0].load(Ordering::Relaxed) + SEEN[1].load(Ordering::Relaxed);
    if all >= EVERY {
        let (fetched, served) = (SEEN[0].swap(0, Ordering::Relaxed), SEEN[1].swap(0, Ordering::Relaxed));
        tracing::info!(
            "fetch ahead (on) over {} landings with changes: {:.1} % merged from early copies with no wait",
            fetched + served,
            100.0 * served as f64 / (fetched + served).max(1) as f64,
        );
    }
}

/// What an early copy holds: by copy (its address), the ranges of offsets
/// and where each lies in the early copy's memory.
#[derive(Default)]
pub(crate) struct Ranges {
    /// By copy, then by the range's start: its end and its place.
    held: HashMap<usize, BTreeMap<usize, (usize, usize)>>,
    bytes: usize,
}

impl Ranges {
    /// Takes `[lo, hi)` of `copy` in: its place, or `None` if a range from
    /// `lo` on is held already.
    pub(crate) fn add(&mut self, copy: usize, lo: usize, hi: usize) -> Option<usize> {
        let ranges = self.held.entry(copy).or_default();
        if lo >= hi || ranges.contains_key(&lo) {
            return None;
        }
        let at = self.bytes;
        ranges.insert(lo, (hi, at));
        self.bytes += hi - lo;
        Some(at)
    }

    /// Where `[lo, hi)` of `copy` lies, if one range holds all of it.
    pub(crate) fn find(&self, copy: usize, lo: usize, hi: usize) -> Option<usize> {
        let (&from, &(to, at)) = self.held.get(&copy)?.range(..=lo).next_back()?;
        (hi <= to).then(|| at + (lo - from))
    }

    /// Bytes held.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Default)]
struct Tally {
    landings: u64,
    covered: u64,
    identical: u64,
    pieces: u64,
    pieces_covered: u64,
    pieces_identical: u64,
    early_bytes: u64,
}

static TALLY: Mutex<Option<Tally>> = Mutex::new(None);

/// One landing's comparison.
#[derive(Default)]
pub(crate) struct Check {
    pieces: u64,
    covered: u64,
    identical: u64,
    early_bytes: u64,
}

impl Check {
    /// An early copy that belongs to this landing.
    pub(crate) fn early(&mut self, ranges: &Ranges) {
        self.early_bytes += ranges.bytes() as u64;
    }

    /// One fetched piece, and the bytes the newest early copy that holds it
    /// has there, if any does.
    pub(crate) fn piece(&mut self, fetched: &[u8], early: Option<&[u8]>) {
        self.pieces += 1;
        if let Some(early) = early {
            self.covered += 1;
            self.identical += u64::from(early == fetched);
        }
    }

    pub(crate) fn finish(self) {
        if self.pieces == 0 {
            return;
        }
        let mut held = TALLY.lock().unwrap_or_else(|e| e.into_inner());
        let t = held.get_or_insert_default();
        t.landings += 1;
        t.covered += u64::from(self.covered == self.pieces);
        t.identical += u64::from(self.identical == self.pieces);
        t.pieces += self.pieces;
        t.pieces_covered += self.covered;
        t.pieces_identical += self.identical;
        t.early_bytes += self.early_bytes;
        if t.landings < EVERY {
            return;
        }
        let percent = |part: u64, whole: u64| 100.0 * part as f64 / whole.max(1) as f64;
        tracing::info!(
            "fetch ahead (verify) over {} landings with a fetch: all pieces held early {:.1} %, all of them \
             identical {:.1} %; pieces held early {:.1} %, of those identical {:.2} %; {:.1} MB copied early \
             per landing",
            t.landings,
            percent(t.covered, t.landings),
            percent(t.identical, t.landings),
            percent(t.pieces_covered, t.pieces),
            percent(t.pieces_identical, t.pieces_covered),
            t.early_bytes as f64 / (1 << 20) as f64 / t.landings as f64,
        );
        *held = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{Ranges, Versions, bound, early_done, early_sent};

    /// An older capture merged after a newer one gets only the bytes the
    /// newer one did not bring.
    #[test]
    fn older_bytes_are_never_merged_over_newer_ones() {
        let mut v = Versions::default();
        assert_eq!(v.claim(1, 100, 200, 5), [(100, 200)]);
        assert_eq!(v.claim(1, 100, 200, 3), [], "older, all held newer");
        assert_eq!(v.claim(1, 50, 250, 4), [(50, 100), (200, 250)], "older, around it");
        assert_eq!(v.claim(1, 0, 300, 4), [(0, 100), (200, 300)], "as new as what lies around");
        assert_eq!(v.claim(1, 150, 160, 7), [(150, 160)]);
        assert_eq!(v.claim(1, 120, 180, 6), [(120, 150), (160, 180)]);
        assert_eq!(v.claim(1, 0, 300, 9), [(0, 300)]);
        assert_eq!(v.claim(1, 0, 300, 8), []);
        assert_eq!(v.claim(2, 0, 300, 1), [(0, 300)], "another copy");
        assert_eq!(v.claim(1, 10, 10, 10), []);
    }

    /// Against a plain array of positions, over many overlapping ranges.
    #[test]
    fn claims_agree_with_a_position_for_every_byte() {
        let mut v = Versions::default();
        let mut plain = [0u64; 64];
        let mut seed = 0x1234_5678_u64;
        let mut next = |n: u64| {
            seed = seed.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
            (seed >> 33) % n
        };
        for _ in 0..4000 {
            let (lo, len, position) = (next(64) as usize, 1 + next(20) as usize, 1 + next(50));
            let hi = (lo + len).min(64);
            let fresh = v.claim(0, lo, hi, position);
            let mut said = [false; 64];
            for (a, b) in fresh {
                assert!(lo <= a && a < b && b <= hi);
                said[a..b].fill(true);
            }
            for i in lo..hi {
                assert_eq!(said[i], plain[i] <= position, "byte {i}");
                plain[i] = plain[i].max(position);
            }
        }
    }

    #[test]
    fn notes_are_kept_for_the_oldest_early_copy_out() {
        assert_eq!(bound(10), 10);
        early_sent(4);
        early_sent(4);
        early_sent(7);
        assert_eq!((bound(10), bound(3)), (4, 3));
        early_done(4);
        assert_eq!(bound(10), 4);
        early_done(4);
        assert_eq!(bound(10), 7);
        early_done(7);
        assert_eq!(bound(10), 10);
    }

    #[test]
    fn early_ranges_are_found_only_whole_and_in_their_copy() {
        let mut ranges = Ranges::default();
        assert_eq!(ranges.add(1, 0x1000, 0x3000), Some(0));
        assert_eq!(ranges.add(1, 0x1000, 0x2000), None, "taken in already");
        assert_eq!(ranges.add(2, 0x1000, 0x2000), Some(0x2000));
        assert_eq!(ranges.add(1, 0x8000, 0x9000), Some(0x3000));
        assert_eq!(ranges.find(1, 0x2000, 0x3000), Some(0x1000));
        assert_eq!(ranges.find(2, 0x1000, 0x2000), Some(0x2000));
        assert_eq!(ranges.find(1, 0x2000, 0x4000), None, "past the end");
        assert_eq!(ranges.find(1, 0x0, 0x1000), None, "before the start");
        assert_eq!(ranges.find(3, 0x1000, 0x2000), None, "another copy");
        assert_eq!(ranges.bytes(), 0x4000);
    }
}
