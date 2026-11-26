//! A measurement, off unless `G6_FETCH_FORECAST=1`: could the pieces of
//! card copies that shaders change be fetched ahead, with the submission
//! that writes them, instead of with one more wait once it has landed
//! (`Host::relay_card_back`)? Nothing is fetched differently.
//!
//! When a submission is sent, each of its pieces is forecast from what was
//! known then: `last`, it had changed the last time it landed; `recent`, in
//! any of the last eight times. When flights land together, the landing
//! would have needed no wait only if every piece that changed in it had
//! been forecast. The log gets the tally every [`EVERY`] landings.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// A copy (its address) and a piece's number in it.
pub(crate) type Piece = (usize, u32);

/// Landings between two lines of the log.
const EVERY: u64 = 200;
/// A piece of a card copy.
const PIECE_KB: u64 = 64;

/// Each piece's last landings, the newest in bit 0: whether it had changed.
static HISTORY: Mutex<Option<HashMap<Piece, u8>>> = Mutex::new(None);
static TALLY: Mutex<Tally> = Mutex::new(Tally::new());

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("G6_FETCH_FORECAST").is_some_and(|v| v == "1") || crate::ahead::copying()
    })
}

/// What one submission's pieces were forecast to do, as it was sent.
pub(crate) struct Forecast(Vec<(Piece, bool, bool)>);

impl Forecast {
    /// The pieces forecast to change by `recent`, the better forecast.
    pub(crate) fn expected(&self) -> impl Iterator<Item = Piece> {
        self.0.iter().filter(|(_, _, recent)| *recent).map(|&(piece, ..)| piece)
    }
}

/// The forecast for a submission with these pieces; `None` if not measuring.
pub(crate) fn at_send(pieces: impl Iterator<Item = Piece>) -> Option<Forecast> {
    on().then(|| {
        let mut history = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
        forecast(history.get_or_insert_default(), pieces)
    })
}

fn forecast(history: &HashMap<Piece, u8>, pieces: impl Iterator<Item = Piece>) -> Forecast {
    let mut seen = HashSet::new();
    Forecast(
        pieces
            .filter(|p| seen.insert(*p))
            .map(|p| {
                let past = history.get(&p).copied().unwrap_or(0);
                (p, past & 1 != 0, past != 0)
            })
            .collect(),
    )
}

/// By the two forecasts (`last`, `recent`).
#[derive(Debug, Default, PartialEq, Eq)]
struct Tally {
    landings: u64,
    /// Landings in which every changed piece had been forecast.
    covered: [u64; 2],
    changed: u64,
    foreseen: [u64; 2],
    /// Pieces that would have been fetched ahead.
    ahead: [u64; 2],
}

impl Tally {
    const fn new() -> Tally {
        Tally { landings: 0, covered: [0; 2], changed: 0, foreseen: [0; 2], ahead: [0; 2] }
    }
}

/// One landing: the plans of the flights landed together.
#[derive(Default)]
pub(crate) struct Landing {
    changed: u64,
    foreseen: [u64; 2],
    ahead: [u64; 2],
    /// What the pieces did, for the history once the landing is complete.
    seen: Vec<(Piece, bool)>,
}

impl Landing {
    /// One plan, with whether the compare flagged each of its pieces.
    pub(crate) fn plan(&mut self, forecast: &Forecast, changed: impl Fn(&Piece) -> bool) {
        for &(piece, last, recent) in &forecast.0 {
            let changed = changed(&piece);
            self.changed += u64::from(changed);
            for (which, said) in [last, recent].into_iter().enumerate() {
                self.ahead[which] += u64::from(said);
                self.foreseen[which] += u64::from(said && changed);
            }
            self.seen.push((piece, changed));
        }
    }

    fn add_to(&self, history: &mut HashMap<Piece, u8>, tally: &mut Tally) {
        for &(piece, changed) in &self.seen {
            let past = history.entry(piece).or_default();
            *past = (*past << 1) | u8::from(changed);
        }
        // A landing with nothing changed waits for nothing as it is.
        if self.changed == 0 {
            return;
        }
        tally.landings += 1;
        tally.changed += self.changed;
        for which in 0..2 {
            tally.covered[which] += u64::from(self.foreseen[which] == self.changed);
            tally.foreseen[which] += self.foreseen[which];
            tally.ahead[which] += self.ahead[which];
        }
    }

    pub(crate) fn finish(self) {
        if self.seen.is_empty() {
            return;
        }
        let mut history = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
        let mut tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
        self.add_to(history.get_or_insert_default(), &mut tally);
        if tally.landings < EVERY {
            return;
        }
        let t = std::mem::take(&mut *tally);
        let percent = |part: u64, whole: u64| 100.0 * part as f64 / whole.max(1) as f64;
        let mb = |pieces: u64| (pieces * PIECE_KB) as f64 / 1024.0 / t.landings as f64;
        tracing::info!(
            "fetch forecast over {} landings with changes: needing no wait {:.1} % (last) {:.1} % (recent); \
             changed pieces foreseen {:.1} % / {:.1} %; per landing {:.1} MB changed pieces, {:.1} / {:.1} MB \
             would be fetched ahead",
            t.landings,
            percent(t.covered[0], t.landings),
            percent(t.covered[1], t.landings),
            percent(t.foreseen[0], t.changed),
            percent(t.foreseen[1], t.changed),
            mb(t.changed),
            mb(t.ahead[0]),
            mb(t.ahead[1]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Piece 1 changes every time, piece 2 every other time: `last` misses
    /// piece 2 when it comes back, `recent` keeps it, at the price of
    /// fetching it when it did not change.
    #[test]
    fn a_landing_is_covered_only_if_every_changed_piece_was_forecast() {
        let (mut history, mut tally) = (HashMap::new(), Tally::new());
        for round in 0..4 {
            let forecast = forecast(&history, [(7, 1), (7, 2), (7, 1)].into_iter());
            let mut landing = Landing::default();
            landing.plan(&forecast, |p| p.1 == 1 || round % 2 == 0);
            landing.add_to(&mut history, &mut tally);
        }
        assert_eq!(
            tally,
            Tally { landings: 4, covered: [2, 3], changed: 6, foreseen: [3, 4], ahead: [5, 6] }
        );
    }
}
