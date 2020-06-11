#![feature(core_intrinsics)]
#![feature(thread_id_value)]
#![feature(stmt_expr_attributes)]

mod profiler;
mod raw_event;
mod serialization;
mod sinks;

use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::convert::Into;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cfg_if::cfg_if;
use fxhash::FxHashMap;
use log::warn;
use parking_lot::RwLock;

use self::serialization::SerializationSink as Sink;

#[inline(never)]
#[cold]
pub fn cold_path<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

#[macro_export]
macro_rules! likely {
    ($e:expr) => {
        #[allow(unused_unsafe)]
        {
            unsafe { std::intrinsics::likely($e) }
        }
    };
}

#[macro_export]
macro_rules! unlikely {
    ($e:expr) => {
        #[allow(unused_unsafe)]
        {
            unsafe { std::intrinsics::unlikely($e) }
        }
    };
}

cfg_if! {
    if #[cfg(windows)] {
        /// FileSerializationSink is faster on Windows
        type SerializationSink = sinks::FileSerializationSink;
    } else if #[cfg(target_arch = "wasm32")] {
        type SerializationSink = sinks::ByteVecSink;
    } else {
        /// MmapSerializatioSink is faster on macOS and Linux
        type SerializationSink = sinks::MmapSerializationSink;
    }
}

type Profiler = profiler::Profiler<SerializationSink>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum ProfileCategory {
    Parsing,
    Lowering,
    Translation,
    Codegen,
    Linking,
    Other,
}

bitflags::bitflags! {
    struct EventFilter: u32 {
        const GENERIC_ACTIVITIES = 1 << 0;
        const QUERY              = 1 << 1;
        const MLIR               = 1 << 2;
        const LLVM               = 1 << 3;

        const DEFAULT = Self::GENERIC_ACTIVITIES.bits | Self::QUERY.bits | Self::MLIR.bits | Self::LLVM.bits;
    }
}

// keep this in sync with the `-Z self-profile-events` help message in liblumen_session/options.rs
const EVENT_FILTERS_BY_NAME: &[(&str, EventFilter)] = &[
    ("none", EventFilter::empty()),
    ("all", EventFilter::all()),
    ("default", EventFilter::DEFAULT),
    ("generic", EventFilter::GENERIC_ACTIVITIES),
    ("query", EventFilter::QUERY),
    ("mlir", EventFilter::MLIR),
    ("llvm", EventFilter::LLVM),
];

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StringId(u32);
impl StringId {
    pub const INVALID: Self = Self(u32::max_value());
}

/// A reference to the SelfProfiler. It can be cloned and sent across thread
/// boundaries at will.
#[derive(Clone)]
pub struct SelfProfilerRef {
    // This field is `None` if self-profiling is disabled for the current
    // compilation session.
    profiler: Option<Arc<SelfProfiler>>,

    // We store the filter mask directly in the reference because that doesn't
    // cost anything and allows for filtering with checking if the profiler is
    // actually enabled.
    event_filter_mask: EventFilter,

    // Print verbose generic activities to stdout
    print_verbose_generic_activities: bool,
}
impl SelfProfilerRef {
    pub fn new(
        profiler: Option<Arc<SelfProfiler>>,
        print_verbose_generic_activities: bool,
    ) -> SelfProfilerRef {
        // If there is no SelfProfiler then the filter mask is set to NONE,
        // ensuring that nothing ever tries to actually access it.
        let event_filter_mask = profiler
            .as_ref()
            .map(|p| p.event_filter_mask)
            .unwrap_or(EventFilter::empty());

        SelfProfilerRef {
            profiler,
            event_filter_mask,
            print_verbose_generic_activities,
        }
    }

    // This shim makes sure that calls only get executed if the filter mask
    // lets them pass. It also contains some trickery to make sure that
    // code is optimized for non-profiling compilation sessions, i.e. anything
    // past the filter check is never inlined so it doesn't clutter the fast
    // path.
    #[inline(always)]
    fn exec<F>(&self, event_filter: EventFilter, f: F) -> TimingGuard<'_>
    where
        F: for<'a> FnOnce(&'a SelfProfiler) -> TimingGuard<'a>,
    {
        #[inline(never)]
        fn cold_call<F>(profiler_ref: &SelfProfilerRef, f: F) -> TimingGuard<'_>
        where
            F: for<'a> FnOnce(&'a SelfProfiler) -> TimingGuard<'a>,
        {
            let profiler = profiler_ref.profiler.as_ref().unwrap();
            f(&**profiler)
        }

        if unlikely!(self.event_filter_mask.contains(event_filter)) {
            cold_call(self, f)
        } else {
            TimingGuard::none()
        }
    }

    /// Start profiling a verbose generic activity. Profiling continues until the
    /// VerboseTimingGuard returned from this call is dropped. In addition to recording
    /// a measureme event, "verbose" generic activities also print a timing entry to
    /// stdout if the compiler is invoked with -Ztime or -Ztime-passes.
    pub fn verbose_generic_activity<'a>(
        &'a self,
        event_label: &'static str,
    ) -> VerboseTimingGuard<'a> {
        let message = if self.print_verbose_generic_activities {
            Some(event_label.to_owned())
        } else {
            None
        };

        VerboseTimingGuard::start(message, self.generic_activity(event_label))
    }

    /// Start profiling a generic activity. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn generic_activity(&self, event_label: &'static str) -> TimingGuard<'_> {
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            TimingGuard::start(profiler, profiler.generic_id, event_label)
        })
    }

    #[inline(always)]
    pub fn query(&self, event_label: &'static str) -> TimingGuard<'_> {
        self.exec(EventFilter::QUERY, |profiler| {
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            TimingGuard::start(profiler, profiler.query_id, event_label)
        })
    }

    #[inline(always)]
    pub fn mlir(&self, event_label: &'static str) -> TimingGuard<'_> {
        self.exec(EventFilter::MLIR, |profiler| {
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            TimingGuard::start(profiler, profiler.mlir_id, event_label)
        })
    }

    #[inline(always)]
    pub fn llvm(&self, event_label: &'static str) -> TimingGuard<'_> {
        self.exec(EventFilter::LLVM, |profiler| {
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            TimingGuard::start(profiler, profiler.llvm_id, event_label)
        })
    }

    pub fn with_profiler(&self, f: impl FnOnce(&SelfProfiler)) {
        if let Some(profiler) = &self.profiler {
            f(&profiler)
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.profiler.is_some()
    }

    #[inline]
    pub fn llvm_recording_enabled(&self) -> bool {
        self.event_filter_mask.contains(EventFilter::LLVM)
    }
    #[inline]
    pub fn get_self_profiler(&self) -> Option<Arc<SelfProfiler>> {
        self.profiler.clone()
    }
}

pub struct SelfProfiler {
    profiler: Profiler,
    event_filter_mask: EventFilter,
    string_cache: RwLock<FxHashMap<String, StringId>>,
    next_string_id: AtomicU32,

    generic_id: StringId,
    query_id: StringId,
    mlir_id: StringId,
    llvm_id: StringId,
}
impl SelfProfiler {
    pub fn new(output_dir: &Path, event_filters: &Option<Vec<String>>) -> anyhow::Result<Self> {
        let sink = Arc::new(SerializationSink::from_path(output_dir)?);
        let profiler = Profiler::new(sink);

        let string_cache = RwLock::new(FxHashMap::default());

        let mut sc = string_cache.write();

        let generic_id = StringId(1);
        sc.insert("generic".to_owned(), generic_id);
        let query_id = StringId(2);
        sc.insert("query".to_owned(), query_id);
        let mlir_id = StringId(3);
        sc.insert("mlir".to_owned(), mlir_id);
        let llvm_id = StringId(4);
        sc.insert("llvm".to_owned(), llvm_id);
        let next_string_id = AtomicU32::new(5);

        drop(sc);

        let mut event_filter_mask = EventFilter::empty();
        if let Some(ref event_filters) = *event_filters {
            let mut unknown_events = vec![];
            for item in event_filters {
                if let Some(&(_, mask)) =
                    EVENT_FILTERS_BY_NAME.iter().find(|&(name, _)| name == item)
                {
                    event_filter_mask |= mask;
                } else {
                    unknown_events.push(item.clone());
                }
            }

            if !unknown_events.is_empty() {
                unknown_events.sort();
                unknown_events.dedup();

                warn!(
                    "Unknown self-profiler events specified: {}. Available options are: {}.",
                    unknown_events.join(", "),
                    EVENT_FILTERS_BY_NAME
                        .iter()
                        .map(|&(name, _)| name.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        } else {
            event_filter_mask = EventFilter::DEFAULT;
        }

        Ok(Self {
            profiler,
            event_filter_mask,
            string_cache,
            next_string_id,
            generic_id,
            query_id,
            mlir_id,
            llvm_id,
        })
    }

    /// Gets a `StringId` for the given string. This method makes sure that
    /// any strings going through it will only be allocated once in the
    /// profiling data.
    pub fn get_or_alloc_cached_string<A>(&self, s: A) -> StringId
    where
        A: Borrow<str> + Into<String>,
    {
        // Only acquire a read-lock first since we assume that the string is
        // already present in the common case.
        {
            let string_cache = self.string_cache.read();

            if let Some(&id) = string_cache.get(s.borrow()) {
                return id;
            }
        }

        let mut string_cache = self.string_cache.write();
        // Check if the string has already been added in the small time window
        // between dropping the read lock and acquiring the write lock.
        match string_cache.entry(s.into()) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let string_id = self.next_string_id.fetch_add(1, Ordering::Relaxed);
                *e.insert(StringId(string_id))
            }
        }
    }
}

#[must_use]
pub struct TimingGuard<'a>(Option<profiler::TimingGuard<'a, SerializationSink>>);

impl<'a> TimingGuard<'a> {
    #[inline]
    pub fn start(
        profiler: &'a SelfProfiler,
        event_kind: StringId,
        event_id: StringId,
    ) -> TimingGuard<'a> {
        let thread_id = std::thread::current().id().as_u64().get() as u32;
        let raw_profiler = &profiler.profiler;
        let timing_guard =
            raw_profiler.start_recording_interval_event(event_kind, event_id, thread_id);
        TimingGuard(Some(timing_guard))
    }

    #[inline]
    pub fn none() -> TimingGuard<'a> {
        TimingGuard(None)
    }

    #[inline(always)]
    pub fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let _timer = self;
        f()
    }
}

#[must_use]
pub struct VerboseTimingGuard<'a> {
    start_and_message: Option<(Instant, String)>,
    _guard: TimingGuard<'a>,
}

impl<'a> VerboseTimingGuard<'a> {
    pub fn start(message: Option<String>, _guard: TimingGuard<'a>) -> Self {
        VerboseTimingGuard {
            _guard,
            start_and_message: message.map(|msg| (Instant::now(), msg)),
        }
    }

    #[inline(always)]
    pub fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let _timer = self;
        f()
    }
}

impl Drop for VerboseTimingGuard<'_> {
    fn drop(&mut self) {
        if let Some((start, ref message)) = self.start_and_message {
            print_time_passes_entry(true, &message[..], start.elapsed());
        }
    }
}

pub fn print_time_passes_entry(do_it: bool, what: &str, dur: Duration) {
    if !do_it {
        return;
    }

    let mem_string = match get_resident() {
        Some(n) => {
            let mb = n as f64 / 1_000_000.0;
            format!("; rss: {}MB", mb.round() as usize)
        }
        None => String::new(),
    };
    println!(
        "time: {}{}\t{}",
        duration_to_secs_str(dur),
        mem_string,
        what
    );
}

// Hack up our own formatting for the duration to make it easier for scripts
// to parse (always use the same number of decimal places and the same unit).
pub fn duration_to_secs_str(dur: std::time::Duration) -> String {
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;
    let secs = dur.as_secs() as f64 + dur.subsec_nanos() as f64 / NANOS_PER_SEC;

    format!("{:.3}", secs)
}

// Memory reporting
cfg_if! {
    if #[cfg(windows)] {
        fn get_resident() -> Option<usize> {
            use std::mem::{self, MaybeUninit};
            use winapi::shared::minwindef::DWORD;
            use winapi::um::processthreadsapi::GetCurrentProcess;
            use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

            let mut pmc = MaybeUninit::<PROCESS_MEMORY_COUNTERS>::uninit();
            match unsafe {
                GetProcessMemoryInfo(GetCurrentProcess(), pmc.as_mut_ptr(), mem::size_of_val(&pmc) as DWORD)
            } {
                0 => None,
                _ => {
                    let pmc = unsafe { pmc.assume_init() };
                    Some(pmc.WorkingSetSize as usize)
                }
            }
        }
    } else if #[cfg(unix)] {
        fn get_resident() -> Option<usize> {
            let field = 1;
            let contents = fs::read("/proc/self/statm").ok()?;
            let contents = String::from_utf8(contents).ok()?;
            let s = contents.split_whitespace().nth(field)?;
            let npages = s.parse::<usize>().ok()?;
            Some(npages * 4096)
        }
    } else {
        fn get_resident() -> Option<usize> {
            None
        }
    }
}
