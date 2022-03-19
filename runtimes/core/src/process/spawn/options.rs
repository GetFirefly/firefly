mod message_queue_data;

use std::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::alloc::{default_heap_size, heap, next_heap_size};
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
    size: usize,
    kill: Option<bool>,
    error_logger: Option<bool>,
}
impl MaxHeapSize {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            kill: None,
            error_logger: None,
        }
    }

    pub fn kill(&mut self, kill: bool) {
        self.kill = Some(kill);
    }

    pub fn error_logger(&mut self, error_logger: bool) {
        self.error_logger = Some(error_logger);
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
    pub fn cascaded_priority(&self, parent_process: Option<&Process>) -> Priority {
        match self.priority {
            Some(priority) => priority,
            None => match parent_process {
                Some(process) => process.priority,
                None => Default::default(),
            },
        }
    }

    pub fn connect(&self, parent_process: Option<&Process>, child_process: &Process) -> Connection {
        let linked = if self.link {
            parent_process.unwrap().link(child_process);

            true
        } else {
            false
        };

        let monitor_reference = if self.monitor {
            let reference = process::monitor(parent_process.unwrap(), child_process);

            Some(reference)
        } else {
            None
        };

        Connection {
            linked,
            monitor_reference,
        }
    }

    pub fn sized_heap(&self) -> Result<(*mut Term, usize), Alloc> {
        let heap_size = self.heap_size().map_err(|_| Alloc::new())?;
        let heap = heap(heap_size)?;

        Ok((heap, heap_size))
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
        let module_function_arity = ModuleFunctionArity {
            module,
            function,
            arity,
        };
        let (heap, heap_size) = self.sized_heap()?;

        let process = Process::new(
            priority,
            parent_process,
            module_function_arity,
            heap,
            heap_size,
        );

        Ok(process)
    }

    // Private

    /// `heap` size in words.
    fn heap_size(&self) -> Result<usize, anyhow::Error> {
        let size = match self.min_heap_size {
            Some(min_heap_size) => next_heap_size(min_heap_size),
            None => default_heap_size(),
        };
        match self.max_heap_size {
            Some(MaxHeapSize {
                size: max_size,
                kill,
                ..
            }) if size >= max_size => {
                if kill.unwrap_or_default() {
                    Err(anyhow!("exceeded maximum heap size of {}", max_size))
                } else {
                    Ok(max_size)
                }
            }
            _ => Ok(size),
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
                "max_heap_size" => {
                    let element = tuple[1];
                    let max_heap_size: Result<usize, _> = element.try_into();
                    if let Ok(max_heap_size) = max_heap_size {
                        self.max_heap_size = Some(MaxHeapSize::new(max_heap_size));
                        return Ok(self);
                    }
                    let max_heap_size_config: Result<Boxed<Map>, _> = element.try_into();
                    match max_heap_size_config {
                        Ok(max_heap_config) => {
                            let max_heap_size_opt = Atom::from_str("max_heap_size");
                            let max_heap_size = max_heap_config
                                .get(max_heap_size_opt.encode().unwrap())
                                .ok_or_else(|| anyhow!("missing max_heap_size key in map"))
                                .and_then(|term| {
                                    term.try_into()
                                        .map_err(|err: TryIntoIntegerError| anyhow!(err))
                                })
                                .context("max_heap_size")?;
                            let mut max_heap_size = MaxHeapSize::new(max_heap_size);
                            let kill_opt = Atom::from_str("kill");
                            if let Some(kill) = max_heap_config.get(kill_opt.encode().unwrap()) {
                                max_heap_size.kill(kill.try_into().context("kill")?);
                            }
                            let error_logger_opt = Atom::from_str("error_logger");
                            if let Some(error_logger) =
                                max_heap_config.get(error_logger_opt.encode().unwrap())
                            {
                                max_heap_size
                                    .error_logger(error_logger.try_into().context("error_logger")?);
                            }
                            self.max_heap_size = Some(max_heap_size);
                            Ok(self)
                        }
                        Err(err) => Err(anyhow!(err).context("max_heap_size")),
                    }
                }
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
     {:max_heap_size, words :: pos_integer() | #{size => non_neg_integer(), kill => boolean(), error_logger => boolean()}}, \
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
