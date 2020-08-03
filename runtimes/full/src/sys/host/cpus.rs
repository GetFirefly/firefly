//! Contains utilities to determine the number of CPUs available on the current system.
//!
//! Sometimes the host will exaggerate the number of CPUs it contains, due to hyperthreading,
//! or similar techniques, employed by the processor. This defines the distinction between
//! logical and physical CPUs/cores.
#![allow(non_snake_case)]
#![allow(dead_code)]

/// Returns the number of physical cores of the current system.
///
/// # Note
///
/// Physical count is supported only on Linux, mac OS and Windows platforms.
/// On other platforms, or if the physical count fails on supported platforms,
/// this function returns the same as [`get()`], which is the number of logical
/// CPUS.
#[inline]
pub fn num_physical() -> usize {
    get_num_physical_cpus()
}

/// Returns the number of available CPUs of the current system.
///
/// This function will get the number of logical cores. Sometimes this is different from the number
/// of physical cores.
///
/// This will check [sched affinity] on Linux, showing a lower number of CPUs if the current
/// thread does not have access to all the computer's CPUs.
///
/// [sched affinity]: http://www.gnu.org/software/libc/manual/html_node/CPU-Affinity.html
#[inline]
pub fn num_logical() -> usize {
    get_num_cpus()
}

/// An alias for the total number of logical CPUs
#[inline]
pub fn num_total() -> usize {
    get_num_cpus()
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
#[inline]
fn get_num_physical_cpus() -> usize {
    // Not implemented on this host, fall back
    get_num_cpus()
}

#[cfg(target_os = "windows")]
fn get_num_physical_cpus() -> usize {
    match get_num_physical_cpus_windows() {
        Some(num) => num,
        None => get_num_cpus(),
    }
}

#[cfg(target_os = "windows")]
fn get_num_physical_cpus_windows() -> Option<usize> {
    // Inspired by https://msdn.microsoft.com/en-us/library/ms683194

    use std::mem;
    use std::ptr;

    #[allow(non_upper_case_globals)]
    const RelationProcessorCore: u32 = 0;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct SYSTEM_LOGICAL_PROCESSOR_INFORMATION {
        mask: usize,
        relationship: u32,
        _unused: [u64; 2],
    }

    extern "system" {
        fn GetLogicalProcessorInformation(
            info: *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
            length: &mut u32,
        ) -> u32;
    }

    // First we need to determine how much space to reserve.

    // The required size of the buffer, in bytes.
    let mut needed_size = 0;

    unsafe {
        GetLogicalProcessorInformation(ptr::null_mut(), &mut needed_size);
    }

    let struct_size = mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>() as u32;

    // Could be 0, or some other bogus size.
    if needed_size == 0 || needed_size < struct_size || needed_size % struct_size != 0 {
        return None;
    }

    let count = needed_size / struct_size;

    // Allocate some memory where we will store the processor info.
    let mut buf = Vec::with_capacity(count as usize);

    let result;

    unsafe {
        result = GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut needed_size);
    }

    // Failed for any reason.
    if result == 0 {
        return None;
    }

    let count = needed_size / struct_size;

    unsafe {
        buf.set_len(count as usize);
    }

    let phys_proc_count = buf.iter()
        // Only interested in processor packages (physical processors.)
        .filter(|proc_info| proc_info.relationship == RelationProcessorCore)
        .count();

    if phys_proc_count == 0 {
        None
    } else {
        Some(phys_proc_count)
    }
}

#[cfg(target_os = "linux")]
fn get_num_physical_cpus() -> usize {
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = match File::open("/proc/cpuinfo") {
        Ok(val) => val,
        Err(_) => return get_num_cpus(),
    };
    let reader = BufReader::new(file);
    let mut set = HashSet::new();
    let mut coreid: u32 = 0;
    let mut physid: u32 = 0;
    let mut chgcount = 0;
    for line in reader.lines().filter_map(|result| result.ok()) {
        let parts: Vec<&str> = line.split(':').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        if parts[0] == "core id" || parts[0] == "physical id" {
            let value = match parts[1].trim().parse() {
                Ok(val) => val,
                Err(_) => break,
            };
            match parts[0] {
                "core id" => coreid = value,
                "physical id" => physid = value,
                _ => {}
            }
            chgcount += 1;
        }
        if chgcount == 2 {
            set.insert((physid, coreid));
            chgcount = 0;
        }
    }
    let count = set.len();
    if count == 0 {
        get_num_cpus()
    } else {
        count
    }
}

#[cfg(windows)]
fn get_num_cpus() -> usize {
    #[repr(C)]
    struct SYSTEM_INFO {
        wProcessorArchitecture: u16,
        wReserved: u16,
        dwPageSize: u32,
        lpMinimumApplicationAddress: *mut u8,
        lpMaximumApplicationAddress: *mut u8,
        dwActiveProcessorMask: *mut u8,
        dwNumberOfProcessors: u32,
        dwProcessorType: u32,
        dwAllocationGranularity: u32,
        wProcessorLevel: u16,
        wProcessorRevision: u16,
    }

    extern "system" {
        fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
    }

    unsafe {
        let mut maybe_uninit_system_info = std::mem::MaybeUninit::uninit();
        GetSystemInfo(maybe_uninit_system_info.as_mut_ptr());
        let system_info = maybe_uninit_system_info.assume_init();

        system_info.dwNumberOfProcessors as usize
    }
}

#[cfg(any(
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "bitrig",
    target_os = "netbsd"
))]
fn get_num_cpus() -> usize {
    let mut cpus: libc::c_uint = 0;
    let mut cpus_size = std::mem::size_of_val(&cpus);

    unsafe {
        cpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as libc::c_uint;
    }
    if cpus < 1 {
        let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];
        unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                2,
                &mut cpus as *mut _ as *mut _,
                &mut cpus_size as *mut _ as *mut _,
                0 as *mut _,
                0,
            );
        }
        if cpus < 1 {
            cpus = 1;
        }
    }
    cpus as usize
}

#[cfg(target_os = "openbsd")]
fn get_num_cpus() -> usize {
    let mut cpus: libc::c_uint = 0;
    let mut cpus_size = std::mem::size_of_val(&cpus);
    let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];

    unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            2,
            &mut cpus as *mut _ as *mut _,
            &mut cpus_size as *mut _ as *mut _,
            0 as *mut _,
            0,
        );
    }
    if cpus < 1 {
        cpus = 1;
    }
    cpus as usize
}

#[cfg(target_os = "macos")]
fn get_num_physical_cpus() -> usize {
    use std::ffi::CStr;
    use std::ptr;

    let mut cpus: i32 = 0;
    let mut cpus_size = std::mem::size_of_val(&cpus);

    let sysctl_name =
        CStr::from_bytes_with_nul(b"hw.physicalcpu\0").expect("byte literal is missing NUL");

    unsafe {
        if 0 != libc::sysctlbyname(
            sysctl_name.as_ptr(),
            &mut cpus as *mut _ as *mut _,
            &mut cpus_size as *mut _ as *mut _,
            ptr::null_mut(),
            0,
        ) {
            return get_num_cpus();
        }
    }
    cpus as usize
}

#[cfg(target_os = "linux")]
fn get_num_cpus() -> usize {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } == 0
    {
        let mut count: u32 = 0;
        for i in 0..libc::CPU_SETSIZE as usize {
            if unsafe { libc::CPU_ISSET(i, &set) } {
                count += 1
            }
        }
        count as usize
    } else {
        let cpus = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if cpus < 1 {
            1
        } else {
            cpus as usize
        }
    }
}

#[cfg(any(
    target_os = "nacl",
    target_os = "macos",
    target_os = "ios",
    target_os = "android",
    target_os = "solaris",
    target_os = "fuchsia"
))]
fn get_num_cpus() -> usize {
    // On ARM targets, processors could be turned off to save power.
    // Use `_SC_NPROCESSORS_CONF` to get the real number.
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    const CONF_NAME: libc::c_int = libc::_SC_NPROCESSORS_CONF;
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    const CONF_NAME: libc::c_int = libc::_SC_NPROCESSORS_ONLN;

    let cpus = unsafe { libc::sysconf(CONF_NAME) };
    if cpus < 1 {
        1
    } else {
        cpus as usize
    }
}

#[cfg(any(target_os = "emscripten", target_os = "redox", target_os = "haiku"))]
fn get_num_cpus() -> usize {
    1
}

#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
fn get_num_cpus() -> usize {
    1
}

#[cfg(test)]
mod tests {
    fn env_var(name: &'static str) -> Option<usize> {
        std::env::var(name).ok().map(|val| val.parse().unwrap())
    }

    #[test]
    fn test_num_logical() {
        let num = super::num_logical();
        if let Some(n) = env_var("NUM_CPUS_TEST_GET") {
            assert_eq!(num, n);
        } else {
            assert!(num > 0);
            assert!(num < 236_451);
        }
    }

    #[test]
    fn test_num_physical() {
        let num = super::num_physical();
        if let Some(n) = env_var("NUM_CPUS_TEST_GET_PHYSICAL") {
            assert_eq!(num, n);
        } else {
            assert!(num > 0);
            assert!(num < 236_451);
        }
    }
}
