//! The share of time the PC's disks are busy, for the window's caption.

use std::sync::Mutex;

use windows_sys::Win32::System::Performance::{
    PDH_FMT_COUNTERVALUE, PDH_FMT_DOUBLE, PdhAddEnglishCounterW, PdhCollectQueryData,
    PdhGetFormattedCounterValue, PdhOpenQueryW,
};

/// The query and its counter, opened at the first asking; `Some(None)` if
/// the system refused them.
static COUNTER: Mutex<Option<Option<(usize, usize)>>> = Mutex::new(None);

fn open() -> Option<(usize, usize)> {
    // Named in English whatever the system's language.
    let path: Vec<u16> = "\\PhysicalDisk(_Total)\\% Idle Time\0".encode_utf16().collect();
    let mut query = std::ptr::null_mut();
    let mut counter = std::ptr::null_mut();
    // SAFETY: our own out pointers and a NUL-terminated path; the query is
    // kept for the life of the process.
    unsafe {
        if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0
            || PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) != 0
            // The first collection is what the next one is measured from.
            || PdhCollectQueryData(query) != 0
        {
            return None;
        }
    }
    Some((query as usize, counter as usize))
}

/// Percent of the time since the last asking that the disks, taken together,
/// were busy. `None` at the first asking, and where the system gives no
/// figure.
pub(crate) fn busy_percent() -> Option<f64> {
    let mut state = COUNTER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(opened) = *state else {
        *state = Some(open());
        return None;
    };
    let (query, counter) = opened?;
    // SAFETY: plain C data, zero is a valid value.
    let mut value: PDH_FMT_COUNTERVALUE = unsafe { std::mem::zeroed() };
    // SAFETY: the query and counter opened above; our own out structure,
    // read as the double that was asked for.
    unsafe {
        if PdhCollectQueryData(query as _) != 0
            || PdhGetFormattedCounterValue(counter as _, PDH_FMT_DOUBLE, std::ptr::null_mut(), &mut value) != 0
        {
            return None;
        }
        Some((100.0 - value.Anonymous.doubleValue).clamp(0.0, 100.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_disks_busy_share_is_a_percentage() {
        let _ = busy_percent();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let share = busy_percent().expect("no figure for the disks");
        assert!((0.0..=100.0).contains(&share), "{share}");
    }
}
