use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::alloc::{default_heap_size, heap, next_heap_size};
use liblumen_alloc::erts::process::{Priority, Process};
use liblumen_alloc::erts::term::{Atom, Boxed, Cons, Term, Tuple, TypedTerm};
use liblumen_alloc::{badarg, ModuleFunctionArity};

use crate::process;

#[must_use]
pub struct Connection {
    pub linked: bool,
    #[must_use]
    pub monitor_reference: Option<Term>,
}

#[derive(Clone, Copy, Debug)]
pub struct MaxHeapSize {
    size: Option<usize>,
    kill: Option<bool>,
    error_logger: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
pub enum MessageQueueData {
    OnHeap,
    OffHeap,
}

impl Default for MessageQueueData {
    fn default() -> Self {
        MessageQueueData::OnHeap
    }
}

impl TryFrom<Term> for MessageQueueData {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let atom: Atom = term.try_into()?;

        match atom.name() {
            "off_heap" => Ok(Self::OffHeap),
            "on_heap" => Ok(Self::OnHeap),
            _ => Err(badarg!().into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub link: bool,
    pub monitor: bool,
    /// When priority is not set it does not default to normal, but instead uses the parent
    /// process's priority
    pub priority: Option<Priority>,
    pub fullsweep_after: Option<usize>,
    pub min_heap_size: Option<usize>,
    pub min_bin_vheap_size: Option<usize>,
    pub max_heap_size: Option<MaxHeapSize>,
    pub message_queue_data: MessageQueueData,
}

impl Options {
    pub fn connect(
        &self,
        parent_process: Option<&Process>,
        child_process: &Process,
    ) -> Result<Connection, Alloc> {
        let linked = if self.link {
            parent_process.unwrap().link(child_process);

            true
        } else {
            false
        };

        let monitor_reference = if self.monitor {
            let reference = process::monitor(parent_process.unwrap(), child_process)?;

            Some(reference)
        } else {
            None
        };

        Ok(Connection {
            linked,
            monitor_reference,
        })
    }

    /// Creates a new process with the memory and priority options.
    ///
    /// To fully apply all options, call `options.connect(&parent_process, &child_process)` after
    /// placing any frames in the `child_process` returns from this function.
    pub fn spawn(
        &self,
        parent_process: Option<&Process>,
        module: Atom,
        function: Atom,
        arity: u8,
    ) -> Result<Process, Alloc> {
        let priority = self.cascaded_priority(parent_process);
        let module_function_arity = Arc::new(ModuleFunctionArity {
            module,
            function,
            arity,
        });
        let (heap, heap_size) = self.sized_heap()?;

        let process = Process::new(
            priority,
            parent_process.map(|process| process.pid()),
            Arc::clone(&module_function_arity),
            heap,
            heap_size,
        );

        Ok(process)
    }

    // Private

    fn cascaded_priority(&self, parent_process: Option<&Process>) -> Priority {
        match self.priority {
            Some(priority) => priority,
            None => match parent_process {
                Some(process) => process.priority,
                None => Default::default(),
            },
        }
    }

    /// `heap` size in words.
    fn heap_size(&self) -> usize {
        match self.min_heap_size {
            Some(min_heap_size) => next_heap_size(min_heap_size),
            None => default_heap_size(),
        }
    }

    fn sized_heap(&self) -> Result<(*mut Term, usize), Alloc> {
        let heap_size = self.heap_size();
        let heap = heap(self.heap_size())?;

        Ok((heap, heap_size))
    }

    fn try_put_option_from_atom(&mut self, atom: Atom) -> bool {
        match atom.name() {
            "link" => {
                self.link = true;

                true
            }
            "monitor" => {
                self.monitor = true;

                true
            }
            _ => false,
        }
    }

    fn try_put_option_from_term(&mut self, term: Term) -> bool {
        match term.to_typed_term().unwrap() {
            TypedTerm::Atom(atom) => self.try_put_option_from_atom(atom),
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Tuple(tuple) => self.try_put_option_from_tuple(&tuple),
                _ => false,
            },
            _ => false,
        }
    }

    fn try_put_option_from_tuple(&mut self, tuple: &Tuple) -> bool {
        tuple.len() == 2 && {
            match tuple[0].to_typed_term().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "fullsweep_after" => match tuple[1].try_into() {
                        Ok(fullsweep_after) => {
                            self.fullsweep_after = Some(fullsweep_after);

                            true
                        }
                        Err(_) => false,
                    },
                    "max_heap_size" => unimplemented!(),
                    "message_queue_data" => match tuple[1].try_into() {
                        Ok(message_queue_data) => {
                            self.message_queue_data = message_queue_data;

                            true
                        }
                        Err(_) => false,
                    },
                    "min_bin_vheap_size" => match tuple[1].try_into() {
                        Ok(min_bin_vheap_size) => {
                            self.min_bin_vheap_size = Some(min_bin_vheap_size);

                            true
                        }
                        Err(_) => false,
                    },
                    "min_heap_size" => match tuple[1].try_into() {
                        Ok(min_heap_size) => {
                            self.min_heap_size = Some(min_heap_size);

                            true
                        }
                        Err(_) => false,
                    },
                    "priority" => match tuple[1].try_into() {
                        Ok(priority) => {
                            self.priority = Some(priority);

                            true
                        }
                        Err(_) => false,
                    },
                    _ => false,
                },
                _ => false,
            }
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            link: false,
            monitor: false,
            priority: None,
            fullsweep_after: None,
            min_heap_size: None,
            min_bin_vheap_size: None,
            max_heap_size: None,
            message_queue_data: Default::default(),
        }
    }
}

impl TryFrom<Boxed<Cons>> for Options {
    type Error = Exception;

    fn try_from(boxed_cons: Boxed<Cons>) -> Result<Self, Self::Error> {
        let mut options: Self = Default::default();
        let mut valid = true;

        for result in boxed_cons.into_iter() {
            valid = match result {
                Ok(element) => options.try_put_option_from_term(element),
                Err(_) => false,
            };

            if !valid {
                break;
            }
        }

        if valid {
            Ok(options)
        } else {
            Err(badarg!().into())
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term.to_typed_term().unwrap() {
            TypedTerm::Nil => Ok(Default::default()),
            TypedTerm::List(cons) => cons.try_into(),
            _ => Err(badarg!().into()),
        }
    }
}
