use alloc::sync::Arc;
use core::num::NonZeroUsize;

use crate::gc::Gc;
use crate::term::*;

use super::monitor::{MonitorFlags, UnaliasMode};
use super::{MaxHeapSize, Priority, Process};

#[derive(Debug, Copy, Clone)]
pub struct MonitorOpts {
    pub alias: Option<UnaliasMode>,
    pub tag: OpaqueTerm,
    pub flags: MonitorFlags,
}
impl Default for MonitorOpts {
    fn default() -> Self {
        Self {
            alias: None,
            tag: OpaqueTerm::NONE,
            flags: MonitorFlags::empty(),
        }
    }
}
impl TryFrom<Term> for MonitorOpts {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Nil => Ok(Self::default()),
            Term::Cons(opts) => {
                let mut monitor_opts = Self::default();
                for result in opts.iter() {
                    let element = result.map_err(|_| ())?;
                    match element.into() {
                        Term::Tuple(pair) if pair.len() == 2 => {
                            let key = pair[0];
                            if !key.is_atom() {
                                return Err(());
                            }
                            let key = key.as_atom();
                            if key == atoms::Alias {
                                let mode = pair[1];
                                if !mode.is_atom() {
                                    return Err(());
                                }
                                let mode: UnaliasMode = mode.as_atom().try_into()?;
                                monitor_opts.alias = Some(mode);
                                monitor_opts.flags |= mode;
                            } else if key == atoms::Tag {
                                monitor_opts.tag = pair[1];
                            } else {
                                return Err(());
                            }
                        }
                        _ => return Err(()),
                    }
                }

                Ok(monitor_opts)
            }
            _ => Err(()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub enum MessageQueueData {
    #[default]
    OffHeap,
    OnHeap,
}
impl TryFrom<Term> for MessageQueueData {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Atom(a) if a == atoms::OffHeap => Ok(Self::OffHeap),
            Term::Atom(a) if a == atoms::OnHeap => Ok(Self::OnHeap),
            _ => Err(()),
        }
    }
}

pub struct Spawned {
    pub process: Arc<Process>,
    pub monitor_ref: Gc<Reference>,
}

pub struct SpawnOpts {
    pub spawn_async: bool,
    pub link: bool,
    pub monitor: Option<MonitorOpts>,
    pub fullsweep_after: Option<usize>,
    pub min_heap_size: Option<NonZeroUsize>,
    pub min_bin_vheap_size: Option<NonZeroUsize>,
    pub max_heap_size: MaxHeapSize,
    pub message_queue_data: MessageQueueData,
    pub priority: Priority,
    pub tag: OpaqueTerm,
}
impl Default for SpawnOpts {
    fn default() -> Self {
        Self {
            spawn_async: false,
            link: false,
            monitor: None,
            fullsweep_after: None,
            min_heap_size: None,
            min_bin_vheap_size: None,
            max_heap_size: Default::default(),
            message_queue_data: Default::default(),
            priority: Default::default(),
            tag: atoms::SpawnReply.into(),
        }
    }
}
impl TryFrom<Term> for SpawnOpts {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Nil => Ok(Default::default()),
            Term::Cons(opts) => {
                let mut spawn_opts = SpawnOpts::default();
                for result in opts.iter() {
                    let element = result.map_err(|_| ())?;
                    match element {
                        Term::Tuple(pair) if pair.len() == 2 => {
                            let key = pair[0];
                            if !key.is_atom() {
                                return Err(());
                            }
                            let key = key.as_atom();
                            let value: Term = pair[1].into();
                            match key {
                                k if k == atoms::Monitor => {
                                    spawn_opts.monitor = Some(value.try_into()?);
                                }
                                k if k == atoms::Priority => {
                                    spawn_opts.priority = value.try_into()?;
                                }
                                k if k == atoms::FullsweepAfter => match value {
                                    Term::Int(i) if i >= 0 => {
                                        spawn_opts.fullsweep_after = Some(i as usize);
                                    }
                                    _ => return Err(()),
                                },
                                k if k == atoms::MinHeapSize => match value {
                                    Term::Int(i) if i >= 0 => {
                                        spawn_opts.min_heap_size = NonZeroUsize::new(i as usize);
                                    }
                                    _ => return Err(()),
                                },
                                k if k == atoms::MinBinVheapSize => match value {
                                    Term::Int(i) if i >= 0 => {
                                        spawn_opts.min_bin_vheap_size =
                                            NonZeroUsize::new(i as usize);
                                    }
                                    _ => return Err(()),
                                },
                                k if k == atoms::MaxHeapSize => {
                                    spawn_opts.max_heap_size = value.try_into()?;
                                }
                                k if k == atoms::MessageQueueData => {
                                    spawn_opts.message_queue_data = value.try_into()?;
                                }
                                _ => return Err(()),
                            }
                        }
                        Term::Atom(a) => {
                            if a == atoms::Link {
                                spawn_opts.link = true;
                            } else if a == atoms::Monitor {
                                spawn_opts.monitor = Some(MonitorOpts::default());
                            } else {
                                return Err(());
                            }
                        }
                        _ => return Err(()),
                    }
                }

                Ok(spawn_opts)
            }
            _ => return Err(()),
        }
    }
}
