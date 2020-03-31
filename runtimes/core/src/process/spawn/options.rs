mod message_queue_data;
mod out_of_code;

use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::alloc::{default_heap_size, heap, next_heap_size};
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::priority::Priority;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use crate::process;
use crate::proplist::TryPropListFromTermError;

use message_queue_data::*;

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
            parent_process,
            Arc::clone(&module_function_arity),
            heap,
            heap_size,
        );
        out_of_code::place_frame_with_arguments(&process, Placement::Push)?;

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

    fn put_option_atom(&mut self, atom: Atom) -> Result<&Self, anyhow::Error> {
        match atom.name() {
            "link" => {
                self.link = true;

                Ok(self)
            }
            "monitor" => {
                self.monitor = true;

                Ok(self)
            }
            name => Err(TryPropListFromTermError::AtomName(name).into()),
        }
    }

    fn put_option_term(&mut self, term: Term) -> Result<&Self, anyhow::Error> {
        match term.decode().unwrap() {
            TypedTerm::Atom(atom) => self.put_option_atom(atom),
            TypedTerm::Tuple(tuple) => self.put_option_tuple(&tuple),
            _ => Err(TryPropListFromTermError::PropertyType.into()),
        }
    }

    fn put_option_tuple(&mut self, tuple: &Tuple) -> Result<&Self, anyhow::Error> {
        if tuple.len() == 2 {
            let atom: Atom = tuple[0]
                .try_into()
                .map_err(|_| TryPropListFromTermError::KeywordKeyType)?;

            match atom.name() {
                "fullsweep_after" => {
                    let fullsweep_after = tuple[1].try_into().context("fullsweep_after")?;
                    self.fullsweep_after = Some(fullsweep_after);

                    Ok(self)
                }
                "max_heap_size" => unimplemented!(),
                "message_queue_data" => {
                    let message_queue_data = tuple[1].try_into().context("message_queue_data")?;
                    self.message_queue_data = message_queue_data;

                    Ok(self)
                }
                "min_bin_vheap_size" => {
                    let min_bin_vheap_size = tuple[1].try_into().context("min_bin_vheap_size")?;
                    self.min_bin_vheap_size = Some(min_bin_vheap_size);

                    Ok(self)
                }
                "min_heap_size" => {
                    let min_heap_size = tuple[1].try_into().context("min_heap_size")?;
                    self.min_heap_size = Some(min_heap_size);

                    Ok(self)
                }
                "priority" => {
                    let priority = tuple[1].try_into().context("priority")?;
                    self.priority = Some(priority);

                    Ok(self)
                }
                name => Err(TryPropListFromTermError::KeywordKeyName(name).into()),
            }
        } else {
            Err(TryPropListFromTermError::TupleNotPair.into())
        }
    }

    fn sized_heap(&self) -> Result<(*mut Term, usize), Alloc> {
        let heap_size = self.heap_size();
        let heap = heap(self.heap_size())?;

        Ok((heap, heap_size))
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

const SUPPORTED_OPTIONS_CONTEXT: &str = "supported options are :link, :monitor, \
     {:fullsweep_after, generational_collections :: pos_integer()}, \
     {:max_heap_size, words :: pos_integer()}, \
     {:message_queue_data, :off_heap | :on_heap}, \
     {:min_bin_vheap_size, words :: pos_integer()}, \
     {:min_heap_size, words :: pos_integer()}, and \
     {:priority, level :: :low | :normal | :high | :max}";

impl TryFrom<Term> for Options {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode().unwrap() {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options
                        .put_option_term(cons.head)
                        .context(SUPPORTED_OPTIONS_CONTEXT)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError).context(SUPPORTED_OPTIONS_CONTEXT),
            };
        }
    }
}
