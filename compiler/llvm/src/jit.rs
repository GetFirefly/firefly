use std::borrow::Borrow;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};

use anyhow::anyhow;

use crate::module::{Module, ModuleRef};
use crate::sys::error::LLVMErrorRef;
use crate::sys::lljit::*;
use crate::sys::orc::*;
use crate::utils::{LLVMString, MemoryBuffer, MemoryBufferRef};

pub trait ObjectCache {
    /// Called prior to compiling a module to see if we already have
    /// the compiled object available.
    ///
    /// Implementations may choose to return a cached object even if the
    /// module has changed, to prevent recompilation, but it is recommended
    /// to return `None` in these cases so that `store` will be called with
    /// the updated object.
    fn get(&self, module: ModuleRef) -> Option<NonNull<crate::sys::LLVMMemoryBuffer>>;
    /// Called after a module has been compiled, with a byte slice representing
    /// the raw object file data to be cached.
    fn store(&mut self, module: ModuleRef, bytes: &[u8]);
}

/// A simple object file cache that caches objects in memory.
pub struct SimpleObjectCache<'c> {
    cache: HashMap<PathBuf, MemoryBuffer<'c>>,
}
impl<'c> SimpleObjectCache<'c> {
    pub fn new() -> Self {
        Self {
            cache: Default::default(),
        }
    }
}
impl<'c> ObjectCache for SimpleObjectCache<'c> {
    fn get(&self, module: ModuleRef) -> Option<NonNull<crate::sys::LLVMMemoryBuffer>> {
        let path = module.source_file();
        self.cache
            .get(path)
            .map(|mb| unsafe { NonNull::new_unchecked(mb.clone().into_raw()) })
    }

    fn store(&mut self, module: ModuleRef, bytes: &[u8]) {
        let path = module.source_file().to_path_buf();
        let mb = MemoryBuffer::create_from_slice(bytes, path.to_str().unwrap());

        self.cache.insert(path, mb);
    }
}

extern "C" {
    type OpaqueObjectCacheWrapper;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct ObjectCacheWrapperRef(*mut OpaqueObjectCacheWrapper);
impl Default for ObjectCacheWrapperRef {
    fn default() -> Self {
        Self(ptr::null::<()>() as *mut OpaqueObjectCacheWrapper)
    }
}

/// Wraps a user-defined object cache implementation so that it can be
/// used as a cache for Orc-based JITs
pub struct ObjectCacheWrapper {
    wrapper: ObjectCacheWrapperRef,
    cache: Box<dyn ObjectCache>,
}
impl ObjectCacheWrapper {
    pub fn new(cache: Box<dyn ObjectCache>) -> Box<Self> {
        let this = Box::new(Self {
            wrapper: Default::default(),
            cache,
        });
        let context = Box::into_raw(this);
        let wrapper = unsafe {
            LLVMObjectCacheWrapperCreate(
                context as *mut c_void,
                object_cache_getter,
                object_cache_notifier,
            )
        };
        let mut this = unsafe { Box::from_raw(context) };
        unsafe { ptr::write(&mut this.wrapper, wrapper) };
        this
    }

    #[inline(always)]
    pub fn as_ref(&self) -> ObjectCacheWrapperRef {
        self.wrapper
    }
}
impl ObjectCache for ObjectCacheWrapper {
    #[inline(always)]
    fn get(&self, module: ModuleRef) -> Option<NonNull<crate::sys::LLVMMemoryBuffer>> {
        self.cache.get(module)
    }

    #[inline(always)]
    fn store(&mut self, module: ModuleRef, bytes: &[u8]) {
        self.cache.store(module, bytes);
    }
}
impl Drop for ObjectCacheWrapper {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeObjectCacheWrapper(self.wrapper);
        }
    }
}

extern "C" fn object_cache_getter(context: *mut c_void, module: ModuleRef) -> MemoryBufferRef {
    let cache = unsafe { &*(context as *const _ as *const ObjectCacheWrapper) };
    cache
        .get(module)
        .map(|nn| nn.as_ptr())
        .unwrap_or_else(|| ptr::null_mut())
}

extern "C" fn object_cache_notifier(
    context: *mut c_void,
    module: ModuleRef,
    ptr: *const u8,
    size: usize,
) {
    let cache = unsafe { &mut *(context as *mut ObjectCacheWrapper) };
    let bytes = unsafe { core::slice::from_raw_parts(ptr, size) };
    cache.store(module, bytes);
}

/// The configuration object for `JitCompiler` instances.
pub struct JitConfig {
    /// When true, enables the object cache, allowing the JIT to avoid
    /// the work of loading an object that has already been loaded.
    enable_caching: bool,
    /// When true, enables the GDB registration listener, which registers
    /// loaded/compiled functions with the debugger (GDB, LLDB)
    enable_gdb_listener: bool,
    /// When true, enables the perf listener, which notifies `perf`
    /// that a function has been emitted
    enable_perf_listener: bool,
    /// When true, will load symbols from the current process into the JIT
    enable_process_symbols: bool,
    /// A set of shared libraries to load when the JIT is instantiated.
    shared_libs: Vec<PathBuf>,
    /// A set of static archives to load when the JIT is instantiated.
    static_libs: Vec<PathBuf>,
}
impl JitConfig {
    /// Create a new, default configuration.
    ///
    /// By default, caching is enabled, debug/perf listeners are enabled,
    /// loading of process symbols is disabled, and no shared libs are
    /// loaded.
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable caching of object files
    #[inline]
    pub fn enable_cache(mut self, enable: bool) -> Self {
        self.enable_caching = enable;
        self
    }

    /// Enable/disable debugging listener
    #[inline]
    pub fn enable_debug_listener(mut self, enable: bool) -> Self {
        self.enable_gdb_listener = enable;
        self
    }

    /// Enable/disable perf listener
    #[inline]
    pub fn enable_perf_listener(mut self, enable: bool) -> Self {
        self.enable_perf_listener = enable;
        self
    }

    /// Enable/disable the loading of symbols from the current process
    #[inline]
    pub fn enable_process_symbols(mut self, enable: bool) -> Self {
        self.enable_process_symbols = enable;
        self
    }

    /// Appends the given path to the set of shared libraries to load
    #[inline]
    pub fn add_shared_library(mut self, lib: PathBuf) -> Self {
        self.shared_libs.push(lib);
        self
    }

    /// Appends the given path to the set of static libraries to load
    #[inline]
    pub fn add_static_library(mut self, lib: PathBuf) -> Self {
        self.static_libs.push(lib);
        self
    }
}
impl Default for JitConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_gdb_listener: true,
            enable_perf_listener: true,
            enable_process_symbols: false,
            shared_libs: vec![],
            static_libs: vec![],
        }
    }
}

// The type of a program's `main` entry point
type MainFn = extern "C" fn(c_int, *const *const u8) -> c_int;

/// A `JitCompiler`, once created, is a fully configured JIT instance, which supports
/// loading object files and LLVM IR modules, and either looking up a symbol address
/// or invoking the loaded program's `main` function.
///
/// See `JitConfig` for configuration options.
pub struct JitCompiler {
    #[allow(unused)]
    object_cache: Option<Box<ObjectCacheWrapper>>,
    jit: LLJIT,
}
impl JitCompiler {
    /// Creates a new instance from the provided configuration.
    ///
    /// If the configuration provided any shared libraries, they will
    /// be loaded during creation.
    pub fn create(config: JitConfig) -> Result<Self, LLVMString> {
        let session = ExecutionSession::new();
        let object_linking_layer = session.create_rtdyld_object_linking_layer();
        let object_linking_layer_ref = object_linking_layer.as_ref();
        object_linking_layer.add_eh_frame_registration_plugin();
        if config.enable_gdb_listener {
            unsafe {
                let listener = LLVMCreateGDBRegistrationListener();
                LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
                    object_linking_layer_ref,
                    listener,
                );
            }
        }
        if config.enable_perf_listener {
            unsafe {
                let listener = LLVMCreatePerfJITEventListener();
                LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
                    object_linking_layer_ref,
                    listener,
                );
            }
        }

        let mut builder = LLJITBuilder::new();
        // We always use a custom ObjectLinkingLayer, since that is
        // where our configuration of the debugging/perf listeners happens
        builder.set_object_linking_layer_creator(
            create_default_object_linking_layer,
            object_linking_layer.into_raw() as *const _ as *mut c_void,
        );

        // Depending on whether or not caching is enabled, we provide our
        // own IRCompiler instance that manages the cache. If not enabled,
        // we use the default IRCompiler created for LLJIT
        let object_cache = if config.enable_caching {
            let simple_cache = Box::new(SimpleObjectCache::new());
            let object_cache = ObjectCacheWrapper::new(simple_cache);
            let object_cache_ptr = object_cache.borrow() as *const _ as *const ObjectCacheWrapper;
            builder.set_compile_function_creator(
                create_default_compile_function,
                object_cache_ptr as *const _ as *mut c_void,
            );
            Some(object_cache)
        } else {
            None
        };

        // Construct the actual LLJIT instance,
        // this takes care of a bunch of details for us that aren't really
        // necessary for us to manage right now, particularly the handling
        // of platform-specific symbols
        let jit = builder.build()?;

        // If there are any shared libs configured, we create dylibs for each
        // one, and associate a definition generator that loads all symbols
        // provided by the given shared library
        let prefix = jit.get_global_prefix();
        for path in config.shared_libs.iter() {
            let c_path = CString::new(path.to_str().unwrap()).unwrap();
            let dylib = session.create_bare_dylib(&c_path);
            dylib.add_generator(DefinitionGenerator::create_dynamic(path, prefix)?);
        }

        // Same for static libs
        for path in config.static_libs.iter() {
            let c_path = CString::new(path.to_str().unwrap()).unwrap();
            let dylib = session.create_bare_dylib(&c_path);
            dylib.add_generator(DefinitionGenerator::create_static(
                path,
                object_linking_layer_ref,
            )?);
        }

        // If loading of symbols from the current process was enabled, then
        // we create a definition generator associated with the main dylib
        if config.enable_process_symbols {
            let prefix = jit.get_global_prefix();
            let main_dylib = jit.get_main_dylib();
            main_dylib.add_generator(DefinitionGenerator::for_current_process(prefix, None)?);
        }

        Ok(Self { object_cache, jit })
    }

    /// Compiles the given LLVM IR module and loads it into the JIT
    pub fn compile(&mut self, module: Module) -> anyhow::Result<()> {
        let session = self.jit.execution_session();
        let dylib_name = CString::new("main").unwrap();
        let dylib = session.get_dylib_by_name(&dylib_name).unwrap();
        let tsm = ThreadSafeModule::create(module, ThreadSafeContext::new());
        self.jit
            .add_module(dylib, tsm)
            .map_err(|err| anyhow!("{}", &err))
    }

    /// Loads an object file from the given path into the JIT
    pub fn load(&mut self, path: &Path) -> anyhow::Result<()> {
        let session = self.jit.execution_session();
        let dylib_name = CString::new(path.to_str().unwrap()).unwrap();
        if let Some(_) = session.get_dylib_by_name(&dylib_name) {
            return Ok(());
        }
        let dylib = session.create_bare_dylib(&dylib_name);
        let buffer = MemoryBuffer::create_from_file(path)?;
        self.jit
            .add_object_file(dylib, buffer)
            .map_err(|err| anyhow!("{}", &err))
    }

    /// Looks up the address of the given symbol name
    #[inline]
    pub fn lookup(&mut self, name: &str) -> Result<*const (), LLVMString> {
        let name = CString::new(name).unwrap();
        self.jit.lookup(&name).map(|addr| addr as *const ())
    }

    /// Invokes the `main` function to run the program loaded in the JIT, if available.
    /// Returns `Ok(status)` if `main` was successfully found, and it returned. Check
    /// the status code to determine if the program itself succeeded.
    ///
    /// If the symbol `main` isn't found, an error will be returned.
    ///
    /// This function blocks until `main` returns.
    pub fn run(&mut self, argv: &[*const u8]) -> Result<i32, LLVMString> {
        let ptr = self.lookup("main")?;
        let main = unsafe { core::mem::transmute::<*const (), MainFn>(ptr) };
        let result = main(argv.len() as c_int, argv.as_ptr());
        Ok(result)
    }
}

extern "C" fn create_default_object_linking_layer(
    context: *const c_void,
    _session: ExecutionSessionRef,
    _triple: *const u8,
) -> LLVMOrcObjectLayerRef {
    context as LLVMOrcObjectLayerRef
}

extern "C" fn create_default_compile_function(
    result: *mut LLVMOrcIRCompilerRef,
    context: *const c_void,
    jtmb: LLVMOrcJITTargetMachineBuilderRef,
) -> LLVMErrorRef {
    let wrapper = unsafe { &*(context as *const ObjectCacheWrapper) };
    unsafe { LLVMOrcCreateOwningCompiler(result, jtmb, wrapper.as_ref()) }
}

pub type ObjectCacheGetObjectFunc = extern "C" fn(*mut c_void, ModuleRef) -> MemoryBufferRef;
pub type ObjectCacheNotifyCompiledFunc = extern "C" fn(*mut c_void, ModuleRef, *const u8, usize);

extern "C" {
    fn LLVMObjectCacheWrapperCreate(
        context: *const c_void,
        getter: ObjectCacheGetObjectFunc,
        notifier: ObjectCacheNotifyCompiledFunc,
    ) -> ObjectCacheWrapperRef;
    fn LLVMDisposeObjectCacheWrapper(cache: ObjectCacheWrapperRef);
    fn LLVMOrcCreateOwningCompiler(
        result: *mut LLVMOrcIRCompilerRef,
        jtmb: LLVMOrcJITTargetMachineBuilderRef,
        cache: ObjectCacheWrapperRef,
    ) -> LLVMErrorRef;
}
