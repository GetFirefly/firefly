macro_rules! badarg {
    ($process:expr, $term:expr) => {
        return {
            $process.exception_info.flags = crate::error::ExceptionFlags::ERROR;
            $process.exception_info.reason = crate::term::atoms::Badarg.into();
            $process.exception_info.value = $term;
            $process.exception_info.args = Some($term);
            $process.exception_info.trace = None;
            $process.exception_info.cause = None;
            crate::function::ErlangResult::Err
        }
    };

    ($process:expr, $term:expr, $cause:expr) => {
        return {
            $process.exception_info.flags = crate::error::ExceptionFlags::ERROR;
            $process.exception_info.reason = crate::term::atoms::Badarg.into();
            $process.exception_info.value = $term;
            $process.exception_info.args = Some($term);
            $process.exception_info.trace = None;
            $process.exception_info.cause = Some($cause);
            crate::function::ErlangResult::Err
        }
    };
}

macro_rules! unwrap_or_badarg {
    ($process:expr, $term:expr, $value:expr) => {
        match $value {
            Ok(value) => value,
            Err(_) => badarg!($process, $term),
        }
    };
}

macro_rules! atom_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Atom(a) => a,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! bool_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Bool(b) => b,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! i64_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Int(i) => i,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! isize_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Int(i) if i >= 0 => match isize::try_from(i) {
                Ok(i) => i,
                Err(_) => {
                    let opaque: OpaqueTerm = Term::Int(i).into();
                    badarg!($process, opaque);
                }
            },
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! usize_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Int(i) if i >= 0 => match usize::try_from(i) {
                Ok(i) => i,
                Err(_) => {
                    let opaque: OpaqueTerm = Term::Int(i).into();
                    badarg!($process, opaque);
                }
            },
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! local_pid_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Pid(pid) if pid.is_local() => i,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! local_reference_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Reference(reference) if reference.is_local() => reference,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! map_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Map(map) => map,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! nonempty_list_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Cons(list) => list,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! tuple_or_badarg {
    ($process:expr, $term:expr) => {
        match $term {
            Term::Tuple(tuple) => tuple,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! tuple_with_arity_or_badarg {
    ($process:expr, $term:expr, $arity:expr) => {
        match $term {
            Term::Tuple(tuple) if tuple.len() == $arity => tuple,
            other => {
                let opaque: OpaqueTerm = other.into();
                badarg!($process, opaque);
            }
        }
    };
}

macro_rules! binary_or_badarg {
    ($process:expr, $term:expr) => {
        if let Some(bin) = $term.as_binary() {
            bin
        } else {
            let opaque: OpaqueTerm = $term.into();
            badarg!($process, opaque);
        }
    };
}

macro_rules! bitstring_or_badarg {
    ($process:expr, $term:expr) => {
        if let Some(bin) = $term.as_bitstring() {
            bin
        } else {
            let opaque: OpaqueTerm = $term.into();
            badarg!($process, opaque);
        }
    };
}

pub mod erlang;
