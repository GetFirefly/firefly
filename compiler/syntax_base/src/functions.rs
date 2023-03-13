use std::fmt;

use firefly_intern::{symbols, Symbol};

use super::types::*;

/// This struct represents either a fully-qualified Erlang function name (i.e. MFA),
/// or a locally-qualified function name (i.e. FA). Fully-qualified function names
/// are module-agnostic (i.e. they are valid in any context), whereas locally-qualified
/// names are context-sensitive in order to resolve them fully.
///
/// A function name when stringified is of the form `M:F/A` or `F/A`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionName {
    pub module: Option<Symbol>,
    pub function: Symbol,
    pub arity: u8,
}
impl FunctionName {
    /// Create a new fully-qualified function name
    pub const fn new(module: Symbol, function: Symbol, arity: u8) -> Self {
        Self {
            module: Some(module),
            function,
            arity,
        }
    }

    /// Create a new locally-qualified function name
    pub const fn new_local(function: Symbol, arity: u8) -> Self {
        Self {
            module: None,
            function,
            arity,
        }
    }

    /// Convert this function name to its locally-qualified form
    #[inline]
    pub fn to_local(mut self) -> Self {
        self.module = None;
        self
    }

    /// Returns true if this represents a locally-qualified name
    #[inline]
    pub fn is_local(&self) -> bool {
        self.module.is_none()
    }

    /// Returns the fully-qualified version of this function name, using the given module
    #[inline]
    pub fn resolve(&self, module: Symbol) -> Self {
        Self {
            module: Some(module),
            function: self.function,
            arity: self.arity,
        }
    }

    pub fn is_bif(&self) -> bool {
        if crate::bifs::get(self).is_some() {
            return true;
        }
        // Special-case handling for match_fail of any arity
        if self.module != Some(symbols::Erlang) {
            return false;
        }
        if self.function == symbols::MatchFail {
            return true;
        }
        false
    }

    pub fn is_guard_bif(&self) -> bool {
        crate::bifs::get(self)
            .map(|sig| sig.visibility.is_guard())
            .unwrap_or_default()
    }

    pub fn is_safe(&self) -> bool {
        if self.module != Some(symbols::Erlang) {
            return false;
        }
        match self.function {
            symbols::NotEqual
            | symbols::NotEqualStrict
            | symbols::Equal
            | symbols::EqualStrict
            | symbols::Lt
            | symbols::Lte
            | symbols::Gt
            | symbols::Gte
                if self.arity == 2 =>
            {
                true
            }
            symbols::Date
            | symbols::Get
            | symbols::GetCookie
            | symbols::GroupLeader
            | symbols::IsAlive
            | symbols::MakeRef
            | symbols::Node
            | symbols::Nodes
            | symbols::Ports
            | symbols::PreLoaded
            | symbols::Processes
            | symbols::Registered
            | symbols::SELF
            | symbols::Time
                if self.arity == 0 =>
            {
                true
            }
            symbols::Get
            | symbols::GetKeys
            | symbols::IsAtom
            | symbols::IsBoolean
            | symbols::IsBinary
            | symbols::IsBitstring
            | symbols::IsFloat
            | symbols::IsFunction
            | symbols::IsInteger
            | symbols::IsList
            | symbols::IsMap
            | symbols::IsNumber
            | symbols::IsPid
            | symbols::IsPort
            | symbols::IsReference
            | symbols::IsTuple
            | symbols::TermToBinary
                if self.arity == 1 =>
            {
                true
            }
            symbols::Max | symbols::Min | symbols::UnpackEnv if self.arity == 2 => true,
            _ => false,
        }
    }

    pub fn is_primop(&self) -> bool {
        if self.module != Some(symbols::Erlang) {
            return false;
        }
        match self.function {
            symbols::NifStart
            | symbols::NifError
            | symbols::MatchFail
            | symbols::Error
            | symbols::Exit
            | symbols::Throw
            | symbols::Halt
            | symbols::Raise
            | symbols::RawRaise
            | symbols::MakeFun
            | symbols::UnpackEnv
            | symbols::BuildStacktrace
            | symbols::BitsInitWritable
            | symbols::RemoveMessage
            | symbols::RecvNext
            | symbols::RecvPeekMessage
            | symbols::RecvWaitTimeout
            | symbols::Yield => true,
            symbols::GarbageCollect if self.arity == 0 => true,
            _ => false,
        }
    }

    pub fn is_exception_op(&self) -> bool {
        if self.module != Some(symbols::Erlang) {
            return false;
        }
        match self.function {
            symbols::MatchFail
            | symbols::NifError
            | symbols::Error
            | symbols::Exit
            | symbols::Throw
            | symbols::Raise
            | symbols::RawRaise => true,
            _ => false,
        }
    }

    /// Returns true if this function name is erlang:apply/2
    pub fn is_apply2(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        self.function == symbols::Apply && self.arity == 2
    }

    pub fn is_type_test(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::IsAtom
            | symbols::IsBinary
            | symbols::IsBitstring
            | symbols::IsBoolean
            | symbols::IsFloat
            | symbols::IsFunction
            | symbols::IsInteger
            | symbols::IsList
            | symbols::IsMap
            | symbols::IsNumber
            | symbols::IsPid
            | symbols::IsPort
            | symbols::IsReference
            | symbols::IsTuple
                if self.arity == 1 =>
            {
                true
            }
            symbols::IsFunction | symbols::IsRecord if self.arity == 2 => true,
            symbols::IsRecord => self.arity == 3,
            _ => false,
        }
    }

    pub fn is_operator(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::Plus | symbols::Minus => self.arity == 1 || self.arity == 2,
            symbols::Bnot => self.arity == 1,
            symbols::Star
            | symbols::Slash
            | symbols::Div
            | symbols::Rem
            | symbols::Band
            | symbols::Bor
            | symbols::Bxor
            | symbols::Bsl
            | symbols::Bsr => self.arity == 2,
            symbols::Not => self.arity == 1,
            symbols::And | symbols::Or | symbols::Xor => self.arity == 2,
            symbols::PlusPlus | symbols::MinusMinus => self.arity == 2,
            symbols::Bang => self.arity == 2,
            symbols::Equal
            | symbols::NotEqual
            | symbols::EqualStrict
            | symbols::NotEqualStrict
            | symbols::Gte
            | symbols::Gt
            | symbols::Lte
            | symbols::Lt => self.arity == 2,
            symbols::Hd => self.arity == 1,
            symbols::Tl => self.arity == 1,
            _ => false,
        }
    }

    pub fn is_arith_op(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::Plus | symbols::Minus => self.arity == 1 || self.arity == 2,
            symbols::Bnot => self.arity == 1,
            symbols::Star
            | symbols::Slash
            | symbols::Div
            | symbols::Rem
            | symbols::Band
            | symbols::Bor
            | symbols::Bxor
            | symbols::Bsl
            | symbols::Bsr => self.arity == 2,
            _ => false,
        }
    }

    pub fn is_bool_op(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::Not => self.arity == 1,
            symbols::And | symbols::Or | symbols::Xor => self.arity == 2,
            _ => false,
        }
    }

    pub fn is_list_op(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::PlusPlus | symbols::MinusMinus => self.arity == 2,
            _ => false,
        }
    }

    pub fn is_send_op(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::Bang => self.arity == 2,
            _ => false,
        }
    }

    pub fn is_comparison_op(&self) -> bool {
        match self.module {
            Some(symbols::Erlang) => (),
            _ => return false,
        }
        match self.function {
            symbols::Equal
            | symbols::NotEqual
            | symbols::EqualStrict
            | symbols::NotEqualStrict
            | symbols::Gte
            | symbols::Gt
            | symbols::Lte
            | symbols::Lt => self.arity == 2,
            _ => false,
        }
    }
}
impl From<&Signature> for FunctionName {
    fn from(sig: &Signature) -> Self {
        Self {
            module: Some(sig.module),
            function: sig.name,
            arity: sig.arity().try_into().unwrap(),
        }
    }
}
impl fmt::Debug for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(module) = self.module {
            write!(f, "{}:{}/{}", module, self.function, self.arity)
        } else {
            write!(f, "{}/{}", self.function, self.arity)
        }
    }
}
impl fmt::Display for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(module) = self.module {
            write!(f, "{}:{}/{}", module, self.function, self.arity)
        } else {
            write!(f, "{}/{}", self.function, self.arity)
        }
    }
}

#[derive(thiserror::Error, Debug, Copy, Clone)]
pub enum InvalidFunctionNameError {
    #[error("expected '/' but reached end of input")]
    MissingArity,
    #[error("invalid arity")]
    InvalidArity,
}
impl std::str::FromStr for FunctionName {
    type Err = InvalidFunctionNameError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (maybe_module, rest) = match s.split_once(':') {
            Some((module, rest)) => (Some(Symbol::intern(module)), rest),
            None => (None, s),
        };
        match rest.rsplit_once('/') {
            Some((function, a)) => {
                let arity: u8 = a
                    .parse()
                    .map_err(|_| InvalidFunctionNameError::InvalidArity)?;
                Ok(FunctionName {
                    module: maybe_module,
                    function: Symbol::intern(function),
                    arity,
                })
            }
            None => Err(InvalidFunctionNameError::MissingArity),
        }
    }
}

bitflags::bitflags! {
    /// There a variety of function types and visibilities in Erlang,
    /// this set of flags can fully express the following:
    ///
    /// * The visibility of a function globally
    /// * Whether the function is imported locally
    /// * Whether the function is referenced locally (i.e. a fully-qualified call is made to it)
    /// * Whether the function is valid in guards
    /// * Whether the function is desirable to inline
    /// * Whether the function was originally defined as a closure
    ///
    /// When lowering into Core, we expand all imported function calls into fully-qualified references,
    /// but it is still useful to know when a function was imported originally
    pub struct Visibility: u8 {
        /// The default consists of no flags, which implies the following:
        ///
        /// * it is private
        /// * it is not imported
        /// * it is not an external reference
        /// * it is not a valid guard
        /// * has not been requested to be inlined
        /// * is not a closure
        /// * is not a nif
        const DEFAULT = 0;
        /// Indicates this function is visibile globally
        const PUBLIC = 1 << 0;
        /// Indicates this function has been aliased locally (i.e. imported)
        /// This also implies the function is referenced locally, so both flags are set if this is set
        const IMPORTED = 1 << 1;
        /// Indicates this function is externally-defined but referenced from the current module
        /// This is always set for imported functions.
        const EXTERNAL = 1 << 2;
        /// Indicates this function is allowed in guards.
        const GUARD = 1 << 3;
        /// Indicates this function can/should be inlined if possible
        /// This is generally set on non-escaping closures, but is also
        /// set on functions which are declared inline via compiler options
        const INLINE = 1 << 4;
        /// Indicates this function is or was defined as a closure
        ///
        /// Closures get initially lifted into module scope during translation, but depending on
        /// their usage, the following may also be true of them:
        ///
        /// * If the closure can be proven to not escape the local module, it can avoid being exported
        /// * If it is non-escaping, it may be inlined and eliminated
        /// * If it escapes, it must be globally visible
        const CLOSURE = 1 << 5;
        /// Indicates this function is defined as a NIF
        const NIF = 1 << 6;
    }
}
impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_imported() {
            f.write_str("import")?;
        } else if self.is_public() {
            f.write_str("public")?;
        } else {
            f.write_str("private")?;
        }
        if self.should_inline() {
            f.write_str(" inline")?;
        }
        if self.is_guard() {
            f.write_str(" guard")?;
        }
        if self.is_nif() {
            f.write_str("nif")?;
        }
        Ok(())
    }
}
impl Visibility {
    /// Returns true if this function is imported from another module or the standard library
    /// prelude
    #[inline(always)]
    pub fn is_imported(&self) -> bool {
        self.contains(Self::IMPORTED)
    }

    /// Returns true if this function's definition does not reside in the current module
    #[inline]
    pub fn is_externally_defined(&self) -> bool {
        self.contains(Self::EXTERNAL) || self.contains(Self::IMPORTED)
    }

    /// Returns true if this function's definition can be found in the current module
    #[inline(always)]
    pub fn is_locally_defined(&self) -> bool {
        !self.is_externally_defined()
    }

    /// Returns true if inlining of this function was requested
    #[inline(always)]
    pub fn should_inline(&self) -> bool {
        self.contains(Self::INLINE)
    }

    /// Returns true if this function is a valid guard function
    #[inline(always)]
    pub fn is_guard(&self) -> bool {
        self.contains(Self::GUARD)
    }

    /// Returns true if this function is declared as a NIF
    #[inline(always)]
    pub fn is_nif(&self) -> bool {
        self.contains(Self::NIF)
    }

    /// Returns true if this function is globally visible
    #[inline(always)]
    pub fn is_public(&self) -> bool {
        self.contains(Self::PUBLIC)
    }

    /// Returns true if this function is only reachable from the current module
    #[inline(always)]
    pub fn is_private(&self) -> bool {
        !self.is_public()
    }
}
impl Default for Visibility {
    #[inline(always)]
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CallConv {
    /// Our calling convention for Erlang uses multi-value returns to propagate an
    /// exception bit and the return value. Additionally, the first argument of an
    /// Erlang function is always a `Process` reference. In all other respects it
    /// follows the C calling convention.
    ///
    /// This is not formally its own calling convention in LLVM, instead
    /// during lowering we transform multi-return values into the best
    /// form for the selected target platform within the bounds of existing
    /// conventions. In the future we plan to define our own calling convention
    /// to do things such as make use of CPU flags for exception state, ensure
    /// tail calls in more scenarios, use multi-return values natively when supported,
    /// etc.
    ///
    /// This is the default calling convention used for Erlang functions
    #[default]
    Erlang,
    /// This is the standard C calling convention for the target platform and is used
    /// for interop with runtime functions implemented with a calling convention different
    /// than the `Erlang` convention. This is primarily used with intrinsics that have different
    /// error handling semantics than `Erlang`, or which return a non-term value.
    ///
    /// NOTE: We use this vs the `Rust` calling convention because `Rust` is not a stable
    /// convention. It is expected that all functions using this convention are implemented
    /// in Rust with `extern "C"` or `extern "C-unwind"`, depending on whether they can panic.
    C,
}
impl fmt::Display for CallConv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Erlang => f.write_str("erlang"),
            Self::C => f.write_str("C"),
        }
    }
}

/// A signature represents the full set of available metadata about a given function.
///
/// Signatures are used both for local definitions, as well as references to externally
/// defined functions. Signatures are useful relative to some module that is being compiled,
/// i.e. the visibility flags encode information about the signature relative to that module
/// specifically, not all modules.
///
/// Signatures can be compared for equality, but when doing so, all of the module-specific
/// contextual information is ignored, and only the function name (i.e. MFA) is considered. The
/// signature types are ignored, as type information is only used for optimization, and in many
/// cases we may not have complete type information anyway.
#[derive(Debug, Clone)]
pub struct Signature {
    pub visibility: Visibility,
    pub cc: CallConv,
    pub module: Symbol,
    pub name: Symbol,
    pub ty: FunctionType,
}
impl Signature {
    pub fn new(
        visibility: Visibility,
        cc: CallConv,
        module: Symbol,
        name: Symbol,
        ty: FunctionType,
    ) -> Self {
        Self {
            visibility,
            cc,
            module,
            name,
            ty,
        }
    }

    /// Used to generate default signatures from an MFA when we don't have explicit
    /// type information available on which to rely.
    ///
    /// The resulting signature describes a globally-visible, external reference to
    /// a function with the given MFA, and using the Erlang calling convention. It
    /// accepts `arity` terms, and returns a term.
    ///
    /// NOTE: The resulting signature should _not_ be used for definitions in the current
    /// module. Those should be created using `new`.
    pub fn generate(name: &FunctionName) -> Self {
        let visibility = Visibility::PUBLIC | Visibility::EXTERNAL;
        let cc = CallConv::Erlang;
        let params = vec![Type::Term(TermType::Any); name.arity as usize];
        let ty = FunctionType {
            results: vec![Type::Term(TermType::Any)],
            params,
        };
        Self {
            visibility,
            cc,
            module: name
                .module
                .expect("generating a signature requires a fully-qualified function name"),
            name: name.function,
            ty,
        }
    }

    /// Returns the fully-qualified function name of this signature
    pub fn mfa(&self) -> FunctionName {
        if self.module == symbols::Empty {
            FunctionName {
                module: None,
                function: self.name,
                arity: self.arity().try_into().unwrap(),
            }
        } else {
            FunctionName {
                module: Some(self.module),
                function: self.name,
                arity: self.arity().try_into().unwrap(),
            }
        }
    }

    /// Returns the arity of the function
    pub fn arity(&self) -> usize {
        self.params().len()
    }

    /// Returns the type signature of this function
    pub fn get_type(&self) -> &FunctionType {
        &self.ty
    }

    /// Returns a slice of the parameter types for this function
    pub fn params(&self) -> &[Type] {
        self.ty.params()
    }

    /// Returns the parameter type of the argument at `index`, if present
    #[inline]
    pub fn param(&self, index: usize) -> Option<&Type> {
        self.ty.params().get(index)
    }

    /// Returns a slice of the result types for this function
    ///
    /// NOTE: This allows for multi-return functions to be expressed, which is leveraged by
    /// the Erlang calling convention. LLVM doesn't actually support multiple returns, so
    /// this ultimately gets lowered to a struct type at that level, but at least one of our
    /// targets supports multi-return natively (WebAssembly), and we want to be able to take
    /// advantage of that when we can.
    pub fn results(&self) -> &[Type] {
        match self.ty.results() {
            [Type::Unit] => &[],
            [Type::Never] => &[],
            results => results,
        }
    }

    /// Returns true if this signature has the given name and arity, disregarding the module
    ///
    /// Put another way, this function answers the question of whether or not this signature
    /// could be the callee for an unqualified function call with the given name and arity.
    pub fn is_local(&self, name: Symbol, arity: u8) -> bool {
        self.name == name && self.arity() == arity as usize
    }

    /// Returns true if this function uses the Erlang calling convention
    pub fn is_erlang(&self) -> bool {
        self.cc == CallConv::Erlang
    }

    /// Returns true if this function was declared as a NIF
    pub fn is_nif(&self) -> bool {
        self.visibility.is_nif()
    }

    /// Returns the calling convention used by this function
    pub fn calling_convention(&self) -> CallConv {
        self.cc
    }

    /// Returns `true` if this function is guaranteed to never return
    pub fn raises(&self) -> bool {
        match self.ty.results() {
            [Type::Never] => true,
            _ => false,
        }
    }
}
impl Eq for Signature {}
impl PartialEq for Signature {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.name == other.name && self.arity() == other.arity()
    }
}
