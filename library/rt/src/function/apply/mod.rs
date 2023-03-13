pub mod dynamic;

#[cfg(feature = "async")]
pub use self::dynamic::DynamicAsyncCallee;
pub use self::dynamic::DynamicCallee;

use core::alloc::Layout;
use core::mem;
use core::slice;

use firefly_arena::DroplessArena;
use firefly_system::sync::OnceLock;

use rustc_hash::FxHasher;

type HashMap<K, V> = hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<FxHasher>>;
type HashSet<V> = hashbrown::HashSet<V, core::hash::BuildHasherDefault<FxHasher>>;

#[cfg(feature = "async")]
use crate::futures::ErlangFuture;
use crate::process::ProcessLock;
use crate::term::{Atom, OpaqueTerm};

use super::{ErlangResult, FunctionSymbol, ModuleFunctionArity};

#[cfg(all(feature = "std", any(unix, windows)))]
const BIFS: &'static [&'static str] = &[
    "erlang:++/2",
    "erlang:--/2",
    "erlang:=:=/2",
    "erlang:==/2",
    "erlang:=/=/2",
    "erlang:/=/2",
    "erlang:>=/2",
    "erlang:=</2",
    "erlang:</2",
    "erlang:>/2",
    "erlang:and/2",
    "erlang:andalso/2",
    "erlang:or/2",
    "erlang:orelse/2",
    "erlang:xor/2",
    "erlang:not/1",
    "erlang:+/2",
    "erlang:-/2",
    "erlang:-/1",
    "erlang:*/2",
    "erlang://2",
    "erlang:div/2",
    "erlang:rem/2",
    "erlang:band/2",
    "erlang:bor/2",
    "erlang:bxor/2",
    "erlang:bsl/2",
    "erlang:bsr/2",
    "erlang:bnot/1",
    "erlang:abs/1",
    "erlang:alias/0",
    "erlang:alias/0",
    "erlang:apply/2",
    "erlang:apply/3",
    "erlang:atom_to_binary/1",
    "erlang:atom_to_binary/2",
    "erlang:atom_to_list/1",
    "erlang:binary_part/2",
    "erlang:binary_part/3",
    "erlang:binary_to_atom/1",
    "erlang:binary_to_atom/2",
    "erlang:binary_to_existing_atom/1",
    "erlang:binary_to_existing_atom/2",
    "erlang:binary_to_float/1",
    "erlang:binary_to_integer/1",
    "erlang:binary_to_integer/2",
    "erlang:binary_to_list/1",
    "erlang:binary_to_list/3",
    "erlang:binary_to_term/1",
    "erlang:binary_to_term/2",
    "erlang:bit_size/1",
    "erlang:bitstring_to_list/1",
    "erlang:byte_size/1",
    "erlang:ceil/1",
    "erlang:date/0",
    "erlang:demonitor/1",
    "erlang:demonitor/2",
    "erlang:disconnect_node/1",
    "erlang:display/1",
    "erlang:element/2",
    "erlang:erase/0",
    "erlang:erase/1",
    "erlang:error/1",
    "erlang:error/2",
    "erlang:error/3",
    "erlang:exit/1",
    "erlang:exit/2",
    "erlang:float/1",
    "erlang:float_to_binary/1",
    "erlang:float_to_binary/2",
    "erlang:float_to_list/1",
    "erlang:float_to_list/2",
    "erlang:floor/1",
    "erlang:garbage_collect/0",
    "erlang:garbage_collect/1",
    "erlang:garbage_collect/2",
    "erlang:get/0",
    "erlang:get/1",
    "erlang:get_keys/0",
    "erlang:get_keys/1",
    "erlang:group_leader/0",
    "erlang:group_leader/2",
    "erlang:halt/0",
    "erlang:halt/1",
    "erlang:halt/2",
    "erlang:hd/1",
    "erlang:integer_to_binary/1",
    "erlang:integer_to_binary/2",
    "erlang:integer_to_list/1",
    "erlang:integer_to_list/2",
    "erlang:iolist_size/1",
    "erlang:iolist_to_binary/1",
    "erlang:is_alive/0",
    "erlang:is_atom/1",
    "erlang:is_binary/1",
    "erlang:is_bitstring/1",
    "erlang:is_boolean/1",
    "erlang:is_float/1",
    "erlang:is_function/1",
    "erlang:is_function/2",
    "erlang:is_integer/1",
    "erlang:is_list/1",
    "erlang:is_map/1",
    "erlang:is_map_key/2",
    "erlang:is_number/1",
    "erlang:is_pid/1",
    "erlang:is_port/1",
    "erlang:is_process_alive/1",
    "erlang:is_record/2",
    "erlang:is_record/3",
    "erlang:is_reference/1",
    "erlang:is_tuple/1",
    "erlang:length/1",
    "erlang:link/1",
    "erlang:list_to_atom/1",
    "erlang:list_to_binary/1",
    "erlang:list_to_bitstring/1",
    "erlang:list_to_existing_atom/1",
    "erlang:list_to_float/1",
    "erlang:list_to_integer/1",
    "erlang:list_to_integer/2",
    "erlang:list_to_pid/1",
    "erlang:list_to_port/1",
    "erlang:list_to_ref/1",
    "erlang:list_to_tuple/1",
    "erlang:load_nif/2",
    "erlang:make_ref/0",
    "erlang:map_get/2",
    "erlang:map_size/1",
    "erlang:max/2",
    "erlang:min/2",
    "erlang:monitor/2",
    "erlang:monitor/3",
    "erlang:monitor_node/2",
    "erlang:monitor_node/3",
    "erlang:node/0",
    "erlang:node/1",
    "erlang:nodes/0",
    "erlang:nodes/1",
    "erlang:now/0",
    "erlang:open_port/2",
    "erlang:pid_to_list/1",
    "erlang:port_close/1",
    "erlang:port_command/2",
    "erlang:port_command/3",
    "erlang:port_connect/2",
    "erlang:port_control/3",
    "erlang:port_to_list/1",
    "erlang:process_flag/2",
    "erlang:process_flag/3",
    "erlang:process_info/1",
    "erlang:process_info/2",
    "erlang:processes/0",
    "erlang:put/2",
    "erlang:raise/2",
    "erlang:raise/3",
    "erlang:ref_to_list/1",
    "erlang:register/2",
    "erlang:registered/0",
    "erlang:round/1",
    "erlang:setelement/3",
    "erlang:self/0",
    "erlang:size/1",
    "erlang:spawn/1",
    "erlang:spawn/2",
    "erlang:spawn/3",
    "erlang:spawn/4",
    "erlang:spawn_link/1",
    "erlang:spawn_link/2",
    "erlang:spawn_link/3",
    "erlang:spawn_link/4",
    "erlang:spawn_monitor/1",
    "erlang:spawn_monitor/2",
    "erlang:spawn_monitor/3",
    "erlang:spawn_monitor/4",
    "erlang:spawn_opt/1",
    "erlang:spawn_opt/2",
    "erlang:spawn_opt/3",
    "erlang:spawn_opt/4",
    "erlang:spawn_opt/5",
    "erlang:spawn_request/1",
    "erlang:spawn_request/2",
    "erlang:spawn_request/3",
    "erlang:spawn_request/4",
    "erlang:spawn_request/5",
    "erlang:spawn_request_abandon/1",
    "erlang:split_binary/2",
    "erlang:statistics/1",
    "erlang:term_to_binary/1",
    "erlang:term_to_binary/2",
    "erlang:term_to_iovec/1",
    "erlang:term_to_iovec/2",
    "erlang:throw/1",
    "erlang:time/0",
    "erlang:tl/1",
    "erlang:trunc/1",
    "erlang:tuple_size/1",
    "erlang:tuple_to_list/1",
    "erlang:unlink/1",
    "erlang:unregister/1",
    "erlang:whereis/1",
    "erlang:yield/0",
];

/// The symbol table used by the runtime system
static SYMBOLS: OnceLock<SymbolTable> = OnceLock::new();

/// Dynamically invokes the function mapped to the given symbol.
///
/// - The caller is responsible for making sure that the given symbol
/// belongs to a function compiled into the executable.
/// - The caller must ensure that the target function adheres to the ABI
/// requirements of the destination function:
///   - C-unwind calling convention
///   - Accepts only immediate-sized terms as arguments
///   - Returns an immediate-sized term as a result
///
/// This function returns `Err` if the given function symbol doesn't exist.
pub fn apply(
    process: &mut ProcessLock,
    symbol: &ModuleFunctionArity,
    args: &[OpaqueTerm],
) -> Result<ErlangResult, ()> {
    if let Some(f) = find_symbol(symbol) {
        Ok(unsafe { dynamic::apply(f, process, args.as_ptr(), args.len()) })
    } else {
        Err(())
    }
}

#[cfg(feature = "async")]
pub fn apply_async(
    process: &mut ProcessLock,
    symbol: &ModuleFunctionArity,
    args: &[OpaqueTerm],
) -> Result<ErlangFuture, ()> {
    if let Some(f) = find_async_symbol(symbol) {
        Ok(unsafe { dynamic::apply_async(f, process, args.as_ptr(), args.len()) })
    } else {
        Err(())
    }
}

pub unsafe fn apply_callee(
    process: &mut ProcessLock,
    callee: DynamicCallee,
    args: &[OpaqueTerm],
) -> ErlangResult {
    dynamic::apply(callee, process, args.as_ptr(), args.len())
}

#[cfg(feature = "async")]
pub unsafe fn apply_callee_async(
    process: &mut ProcessLock,
    callee: DynamicAsyncCallee,
    args: &[OpaqueTerm],
) -> ErlangFuture {
    dynamic::apply_async(callee, process, args.as_ptr(), args.len())
}

pub fn find_symbol(mfa: &ModuleFunctionArity) -> Option<DynamicCallee> {
    if let Some(f) = SYMBOLS.get().and_then(|table| table.get_function(mfa)) {
        Some(unsafe { mem::transmute::<*const (), DynamicCallee>(f) })
    } else {
        None
    }
}

#[cfg(feature = "async")]
pub fn find_async_symbol(mfa: &ModuleFunctionArity) -> Option<DynamicAsyncCallee> {
    if let Some(f) = SYMBOLS.get().and_then(|table| table.get_function(mfa)) {
        Some(unsafe { mem::transmute::<*const (), DynamicAsyncCallee>(f) })
    } else {
        None
    }
}

#[cfg(all(feature = "std", any(unix, windows)))]
pub fn find_native_symbol<T>(symbol: &[u8]) -> Result<Symbol<T>, libloading::Error> {
    SYMBOLS.get().unwrap().get_symbol(symbol)
}

pub fn module_loaded(module: Atom) -> bool {
    SYMBOLS
        .get()
        .map(|table| table.contains_module(module))
        .unwrap_or(false)
}

/// Performs one-time initialization of the atom table at program start, using the
/// array of constant atom values present in the compiled program.
///
/// It is expected that this will be called by code generated by the compiler, during the
/// earliest phase of startup, to ensure that nothing has tried to use the atom table yet.
#[export_name = "__firefly_initialize_dispatch_table"]
pub unsafe extern "C-unwind" fn init(
    start: *const FunctionSymbol,
    end: *const FunctionSymbol,
) -> bool {
    if start == end {
        if let Err(_) = SYMBOLS.set(SymbolTable::new(0)) {
            panic!("tried to initialize dispatch table more than once!");
        }
        return true;
    }
    if start.is_null() || end.is_null() {
        return false;
    }

    debug_assert_eq!(
        ((end as usize) - (start as usize)) % mem::size_of::<FunctionSymbol>(),
        0,
        "invalid function symbol range"
    );

    let len = end.offset_from(start).try_into().unwrap();
    let data = slice::from_raw_parts::<'static, _>(start, len);

    let mut table = SymbolTable::new(len);
    for symbol in data.iter().copied() {
        let module = symbol.module;
        let function = symbol.function;
        let arity = symbol.arity;
        let callee = symbol.ptr;

        // If the callee pointer is null, then this was a weakly linked symbol,
        // and the native implementation is not available.
        if callee.is_null() {
            continue;
        }

        let layout = Layout::new::<ModuleFunctionArity>();
        let ptr = table.arena.alloc_raw(layout) as *mut ModuleFunctionArity;
        ptr.write(ModuleFunctionArity {
            module,
            function,
            arity,
        });
        let sym = mem::transmute::<&ModuleFunctionArity, &'static ModuleFunctionArity>(&*ptr);
        assert_eq!(None, table.idents.insert(callee, sym));
        assert_eq!(None, table.functions.insert(sym, callee));
        table.modules.insert(sym.module);
    }

    if let Err(_) = SYMBOLS.set(table) {
        panic!("tried to initialize dispatch table more than once!");
    }

    true
}

#[cfg(all(feature = "std", unix))]
type Library = libloading::os::unix::Library;
#[cfg(all(feature = "std", windows))]
type Library = libloading::os::windows::Library;

#[cfg(all(feature = "std", unix))]
pub type Symbol<T> = libloading::os::unix::Symbol<T>;
#[cfg(all(feature = "std", windows))]
pub type Symbol<T> = libloading::os::windows::Symbol<T>;

struct SymbolTable {
    #[cfg(all(feature = "std", any(unix, windows)))]
    library: Library,
    functions: HashMap<&'static ModuleFunctionArity, *const ()>,
    idents: HashMap<*const (), &'static ModuleFunctionArity>,
    modules: HashSet<Atom>,
    arena: DroplessArena,
}
impl SymbolTable {
    #[cfg(all(feature = "std", any(unix, windows)))]
    fn new(size: usize) -> Self {
        let library = Library::this().into();
        Self {
            library,
            functions: HashMap::with_capacity_and_hasher(BIFS.len() + size, Default::default()),
            idents: HashMap::with_capacity_and_hasher(BIFS.len() + size, Default::default()),
            modules: HashSet::default(),
            arena: DroplessArena::default(),
        }
    }

    #[cfg(not(all(feature = "std", any(unix, windows))))]
    fn new(size: usize) -> Self {
        Self {
            functions: HashMap::with_capacity_and_hasher(size, Default::default()),
            idents: HashMap::with_capacity_and_hasher(size, Default::default()),
            modules: HashSet::default(),
            arena: DroplessArena::default(),
        }
    }

    #[cfg(all(feature = "std", any(unix, windows)))]
    #[allow(unused)]
    fn fill(&mut self) {
        use core::ops::Deref;

        unsafe {
            for bif in BIFS.iter().copied() {
                let mfa = bif.parse::<ModuleFunctionArity>().unwrap();
                if self.functions.contains_key(&mfa) {
                    continue;
                }
                let sym: Result<Symbol<unsafe extern "C" fn() -> ()>, _> =
                    self.library.get(bif.as_bytes());
                if let Ok(sym) = sym {
                    let callee = *sym.deref() as *const ();
                    let layout = Layout::new::<ModuleFunctionArity>();
                    let ptr = self.arena.alloc_raw(layout) as *mut ModuleFunctionArity;
                    ptr.write(mfa);
                    let sym =
                        mem::transmute::<&ModuleFunctionArity, &'static ModuleFunctionArity>(&*ptr);
                    assert_eq!(None, self.idents.insert(callee, sym));
                    assert_eq!(None, self.functions.insert(sym, callee));
                    self.modules.insert(sym.module);
                }
            }
        }
    }

    #[cfg(not(all(feature = "std", any(unix, windows))))]
    #[allow(unused)]
    fn fill(&mut self) {}

    #[cfg(all(feature = "std", any(unix, windows)))]
    fn get_symbol<T>(&self, symbol: &[u8]) -> Result<Symbol<T>, libloading::Error> {
        unsafe { self.library.get(symbol) }
    }

    #[allow(unused)]
    fn get_ident(&self, function: *const ()) -> Option<&'static ModuleFunctionArity> {
        self.idents.get(&function).copied()
    }

    fn get_function(&self, ident: &ModuleFunctionArity) -> Option<*const ()> {
        self.functions.get(ident).copied()
    }

    fn contains_module(&self, module: Atom) -> bool {
        self.modules.contains(&module)
    }
}
impl Default for SymbolTable {
    fn default() -> Self {
        SymbolTable::new(100)
    }
}

// These are safe to implement because the items in the symbol table are static
unsafe impl Sync for SymbolTable {}
unsafe impl Send for SymbolTable {}
