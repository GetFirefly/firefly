use crate::spec::{Target, TargetOptions};

// See https://developer.android.com/ndk/guides/abis.html#arm64-v8a
// for target ABI requirements.

pub fn target() -> Target {
    Target {
        llvm_target: "aarch64-linux-android".into(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            max_atomic_width: Some(128),
            // As documented in http://developer.android.com/ndk/guides/cpu-features.html
            // the neon (ASIMD) and FP must exist on all android aarch64 targets.
            features: "+neon,+fp-armv8".into(),

            ..super::android_base::opts()
        },
    }
}
