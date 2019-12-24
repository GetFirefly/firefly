macro_rules! term_try_into_atom {
    ($name:ident) => {
        crate::context::term_try_into_atom(stringify!($name), $name)
    };
}

macro_rules! term_try_into_isize {
    ($name:ident) => {
        crate::context::term_try_into_isize(stringify!($name), $name)
    };
}

macro_rules! term_try_into_local_reference {
    ($name:ident) => {
        crate::context::term_try_into_local_reference(stringify!($name), $name)
    };
}

macro_rules! term_try_into_map_or_badmap {
    ($process:expr, $name:ident) => {
        crate::context::term_try_into_map_or_badmap($process, stringify!($name), $name)
    };
}

macro_rules! term_try_into_non_empty_list {
    ($name:ident) => {
        crate::context::term_try_into_non_empty_list(stringify!($name), $name)
    };
}

macro_rules! term_try_into_time_unit {
    ($name:ident) => {
        crate::context::term_try_into_time_unit(stringify!($name), $name)
    };
}

macro_rules! term_try_into_tuple {
    ($name:ident) => {
        crate::context::term_try_into_tuple(stringify!($name), $name)
    };
}
