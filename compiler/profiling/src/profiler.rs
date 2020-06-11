use std::sync::Arc;
use std::time::Instant;

use crate::raw_event::RawEvent;
use crate::serialization::SerializationSink;
use crate::StringId;

pub struct Profiler<S: SerializationSink> {
    event_sink: Arc<S>,
    start_time: Instant,
}
impl<S: SerializationSink> Profiler<S> {
    pub fn new(event_sink: Arc<S>) -> Self {
        Self {
            event_sink,
            start_time: Instant::now(),
        }
    }

    /// Creates a "start" event and returns a `TimingGuard` that will create
    /// the corresponding "end" event when it is dropped.
    #[inline]
    pub fn start_recording_interval_event<'a>(
        &'a self,
        event_kind: StringId,
        event_id: StringId,
        thread_id: u32,
    ) -> TimingGuard<'a, S> {
        TimingGuard {
            profiler: self,
            event_id,
            event_kind,
            thread_id,
            start_ns: self.nanos_since_start(),
        }
    }

    fn record_raw_event(&self, raw_event: &RawEvent) {
        self.event_sink
            .write_atomic(std::mem::size_of::<RawEvent>(), |bytes| {
                raw_event.serialize(bytes);
            });
    }

    fn nanos_since_start(&self) -> u64 {
        let duration_since_start = self.start_time.elapsed();
        duration_since_start.as_secs() * 1_000_000_000 + duration_since_start.subsec_nanos() as u64
    }
}

/// When dropped, this `TimingGuard` will record an "end" event in the
/// `Profiler` it was created by.
#[must_use]
pub struct TimingGuard<'a, S: SerializationSink> {
    profiler: &'a Profiler<S>,
    event_id: StringId,
    event_kind: StringId,
    thread_id: u32,
    start_ns: u64,
}

impl<'a, S: SerializationSink> Drop for TimingGuard<'a, S> {
    #[inline]
    fn drop(&mut self) {
        let raw_event = RawEvent::new_interval(
            self.event_kind,
            self.event_id,
            self.thread_id,
            self.start_ns,
            self.profiler.nanos_since_start(),
        );

        self.profiler.record_raw_event(&raw_event);
    }
}
