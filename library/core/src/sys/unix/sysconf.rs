#[inline]
pub fn pagesize() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[allow(unused)]
#[cfg(any(
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "bitrig",
    target_os = "netbsd"
))]
pub fn get_num_cpus() -> usize {
    use core::mem;

    let mut cpus: libc::c_uint = 0;
    let mut cpus_size = mem::size_of_val(&cpus);

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

#[allow(unused)]
#[cfg(target_os = "openbsd")]
pub fn get_num_cpus() -> usize {
    use core::mem;

    let mut cpus: libc::c_uint = 0;
    let mut cpus_size = mem::size_of_val(&cpus);
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

#[allow(unused)]
#[cfg(target_os = "linux")]
pub fn get_num_cpus() -> usize {
    use core::mem;

    let mut set: libc::cpu_set_t = unsafe { mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mut set) } == 0 {
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

#[allow(unused)]
#[cfg(any(
    target_os = "nacl",
    target_os = "macos",
    target_os = "ios",
    target_os = "android",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "fuchsia"
))]
pub fn get_num_cpus() -> usize {
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

#[allow(unused)]
#[cfg(not(any(
    target_os = "nacl",
    target_os = "macos",
    target_os = "ios",
    target_os = "android",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "fuchsia",
    target_os = "linux",
    target_os = "openbsd",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "bitrig",
    target_os = "netbsd",
)))]
pub fn get_num_cpus() -> usize {
    1
}
