use crate::StringId;

#[derive(Eq, PartialEq, Debug)]
#[repr(C)]
pub struct RawEvent {
    pub event_kind: StringId,
    pub event_id: StringId,
    pub thread_id: u32,

    // The following 96 bits store the start and the end timestamp, using
    // 48 bits for each.
    pub start_time_lower: u32,
    pub end_time_lower: u32,
    pub start_and_end_upper: u32,
}

/// `RawEvents` that have an end time stamp with this value are instant events.
const INSTANT_TIMESTAMP_MARKER: u64 = 0xFFFF_FFFF_FFFF;

/// The max interval timestamp we can represent with the 48 bits available.
/// The highest value is reserved for the `INSTANT_TIMESTAMP_MARKER`.
pub const MAX_INTERVAL_TIMESTAMP: u64 = INSTANT_TIMESTAMP_MARKER - 1;

impl RawEvent {
    #[inline]
    pub fn new_interval(
        event_kind: StringId,
        event_id: StringId,
        thread_id: u32,
        start_nanos: u64,
        end_nanos: u64,
    ) -> RawEvent {
        assert!(start_nanos <= end_nanos);
        assert!(end_nanos <= MAX_INTERVAL_TIMESTAMP);

        let start_time_lower = start_nanos as u32;
        let end_time_lower = end_nanos as u32;

        let start_time_upper = (start_nanos >> 16) as u32 & 0xFFFF_0000;
        let end_time_upper = (end_nanos >> 32) as u32;

        let start_and_end_upper = start_time_upper | end_time_upper;

        RawEvent {
            event_kind,
            event_id,
            thread_id,
            start_time_lower,
            end_time_lower,
            start_and_end_upper,
        }
    }

    #[inline]
    pub fn start_nanos(&self) -> u64 {
        self.start_time_lower as u64 | (((self.start_and_end_upper & 0xFFFF_0000) as u64) << 16)
    }

    #[inline]
    pub fn end_nanos(&self) -> u64 {
        self.end_time_lower as u64 | (((self.start_and_end_upper & 0x0000_FFFF) as u64) << 32)
    }

    #[inline]
    pub fn serialize(&self, bytes: &mut [u8]) {
        assert!(bytes.len() == std::mem::size_of::<RawEvent>());

        #[cfg(target_endian = "little")]
        {
            let raw_event_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self as *const _ as *const u8,
                    std::mem::size_of::<RawEvent>(),
                )
            };

            bytes.copy_from_slice(raw_event_bytes);
        }

        #[cfg(target_endian = "big")]
        {
            // We always emit data as little endian, which we have to do
            // manually on big endian targets.
            use byteorder::{ByteOrder, LittleEndian};

            LittleEndian::write_u32(&mut bytes[0..], self.event_kind.as_u32());
            LittleEndian::write_u32(&mut bytes[4..], self.event_id.as_u32());
            LittleEndian::write_u32(&mut bytes[8..], self.thread_id);
            LittleEndian::write_u32(&mut bytes[12..], self.start_time_lower);
            LittleEndian::write_u32(&mut bytes[16..], self.end_time_lower);
            LittleEndian::write_u32(&mut bytes[20..], self.start_and_end_upper);
        }
    }

    #[inline]
    pub fn deserialize(bytes: &[u8]) -> RawEvent {
        assert!(bytes.len() == std::mem::size_of::<RawEvent>());

        #[cfg(target_endian = "little")]
        {
            let mut raw_event = RawEvent::default();
            unsafe {
                let raw_event = std::slice::from_raw_parts_mut(
                    &mut raw_event as *mut RawEvent as *mut u8,
                    std::mem::size_of::<RawEvent>(),
                );
                raw_event.copy_from_slice(bytes);
            };
            raw_event
        }

        #[cfg(target_endian = "big")]
        {
            use byteorder::{ByteOrder, LittleEndian};
            RawEvent {
                event_kind: StringId::new(LittleEndian::read_u32(&bytes[0..])),
                event_id: StringId::from_u32(LittleEndian::read_u32(&bytes[4..])),
                thread_id: LittleEndian::read_u32(&bytes[8..]),
                start_time_lower: LittleEndian::read_u32(&bytes[12..]),
                end_time_lower: LittleEndian::read_u32(&bytes[16..]),
                start_and_end_upper: LittleEndian::read_u32(&bytes[20..]),
            }
        }
    }
}

impl Default for RawEvent {
    fn default() -> Self {
        RawEvent {
            event_kind: StringId::INVALID,
            event_id: StringId::INVALID,
            thread_id: 0,
            start_time_lower: 0,
            end_time_lower: 0,
            start_and_end_upper: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_event_has_expected_size() {
        // A test case to prevent accidental regressions of RawEvent's size.
        assert_eq!(std::mem::size_of::<RawEvent>(), 24);
    }

    #[test]
    fn is_instant() {
        assert!(RawEvent::new_instant(StringId::INVALID, StringId::INVALID, 987, 0,).is_instant());

        assert!(RawEvent::new_instant(
            StringId::INVALID,
            StringId::INVALID,
            987,
            MAX_INSTANT_TIMESTAMP,
        )
        .is_instant());

        assert!(!RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            987,
            0,
            MAX_INTERVAL_TIMESTAMP,
        )
        .is_instant());
    }

    #[test]
    #[should_panic]
    fn invalid_instant_timestamp() {
        let _ = RawEvent::new_instant(
            StringId::INVALID,
            StringId::INVALID,
            123,
            // timestamp too large
            MAX_INSTANT_TIMESTAMP + 1,
        );
    }

    #[test]
    #[should_panic]
    fn invalid_start_timestamp() {
        let _ = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            123,
            // start timestamp too large
            MAX_INTERVAL_TIMESTAMP + 1,
            MAX_INTERVAL_TIMESTAMP + 1,
        );
    }

    #[test]
    #[should_panic]
    fn invalid_end_timestamp() {
        let _ = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            123,
            0,
            // end timestamp too large
            MAX_INTERVAL_TIMESTAMP + 3,
        );
    }

    #[test]
    #[should_panic]
    fn invalid_end_timestamp2() {
        let _ = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            123,
            0,
            INSTANT_TIMESTAMP_MARKER,
        );
    }

    #[test]
    #[should_panic]
    fn start_greater_than_end_timestamp() {
        let _ = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            123,
            // start timestamp greater than end timestamp
            1,
            0,
        );
    }

    #[test]
    fn start_equal_to_end_timestamp() {
        // This is allowed, make sure we don't panic
        let _ = RawEvent::new_interval(StringId::INVALID, StringId::INVALID, 123, 1, 1);
    }

    #[test]
    fn interval_timestamp_decoding() {
        // Check the upper limits
        let e = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            1234,
            MAX_INTERVAL_TIMESTAMP,
            MAX_INTERVAL_TIMESTAMP,
        );

        assert_eq!(e.start_nanos(), MAX_INTERVAL_TIMESTAMP);
        assert_eq!(e.end_nanos(), MAX_INTERVAL_TIMESTAMP);

        // Check the lower limits
        let e = RawEvent::new_interval(StringId::INVALID, StringId::INVALID, 1234, 0, 0);

        assert_eq!(e.start_nanos(), 0);
        assert_eq!(e.end_nanos(), 0);

        // Check that end does not bleed into start
        let e = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            1234,
            0,
            MAX_INTERVAL_TIMESTAMP,
        );

        assert_eq!(e.start_nanos(), 0);
        assert_eq!(e.end_nanos(), MAX_INTERVAL_TIMESTAMP);

        // Test some random values
        let e = RawEvent::new_interval(
            StringId::INVALID,
            StringId::INVALID,
            1234,
            0x1234567890,
            0x1234567890A,
        );

        assert_eq!(e.start_nanos(), 0x1234567890);
        assert_eq!(e.end_nanos(), 0x1234567890A);
    }

    #[test]
    fn instant_timestamp_decoding() {
        assert_eq!(
            RawEvent::new_instant(StringId::INVALID, StringId::INVALID, 987, 0,).start_nanos(),
            0
        );

        assert_eq!(
            RawEvent::new_instant(
                StringId::INVALID,
                StringId::INVALID,
                987,
                MAX_INSTANT_TIMESTAMP,
            )
            .start_nanos(),
            MAX_INSTANT_TIMESTAMP
        );
    }
}
