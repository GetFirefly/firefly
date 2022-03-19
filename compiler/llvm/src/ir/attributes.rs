use std::mem::MaybeUninit;

use paste::paste;

use super::*;
use crate::support::*;

extern "C" {
    type LlvmAttribute;
}

/// This trait is implemented by all attributes
pub trait Attribute {
    fn is_enum_attribute(&self) -> bool {
        extern "C" {
            fn LLVMIsEnumAttribute(attr: AttributeBase) -> bool;
        }
        unsafe { LLVMIsEnumAttribute(self.base()) }
    }

    fn is_string_attribute(&self) -> bool {
        extern "C" {
            fn LLVMIsStringAttribute(attr: AttributeBase) -> bool;
        }
        unsafe { LLVMIsStringAttribute(self.base()) }
    }

    fn is_type_attribute(&self) -> bool {
        extern "C" {
            fn LLVMIsTypeAttribute(attr: AttributeBase) -> bool;
        }
        unsafe { LLVMIsTypeAttribute(self.base()) }
    }

    /// Adds this attribute to the given function, at the specified place
    fn add(&self, fun: Function, index: AttributePlace) {
        extern "C" {
            fn LLVMAddAttributeAtIndex(fun: Function, index: u32, attr: AttributeBase);
        }
        unsafe { LLVMAddAttributeAtIndex(fun, index.into(), self.base()) }
    }

    /// Removes this attribute from the given function and place
    fn remove(&self, fun: Function, index: AttributePlace);

    /// Returns an opaque handle to the underlying attribute
    fn base(&self) -> AttributeBase;
}

/// Represents the base type for all attributes in LLVM IR
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AttributeBase(*const LlvmAttribute);
impl Attribute for AttributeBase {
    fn remove(&self, fun: Function, index: AttributePlace) {
        if let Ok(attr) = EnumAttribute::try_from(*self) {
            attr.remove(fun, index);
        } else if let Ok(attr) = TypeAttribute::try_from(*self) {
            attr.remove(fun, index);
        } else {
            let attr = StringAttribute::try_from(*self).unwrap();
            attr.remove(fun, index);
        }
    }

    #[inline(always)]
    fn base(&self) -> AttributeBase {
        *self
    }
}
impl AttributeBase {
    pub const fn null() -> Self {
        let ptr =
            unsafe { std::mem::transmute::<*const (), *const LlvmAttribute>(std::ptr::null()) };
        Self(ptr)
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

macro_rules! impl_attr_traits {
    ($ty:ident, $kind:ident) => {
        paste! {
            impl_attr_traits!($ty, $kind, [<is_ $kind _attribute>]);
        }
    };

    ($ty:ident, $kind:ident, $kind_predicate:ident) => {
        impl Into<AttributeBase> for $ty {
            fn into(self) -> AttributeBase {
                self.0
            }
        }
        impl TryFrom<AttributeBase> for $ty {
            type Error = InvalidTypeCastError;
            fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
                if attr.$kind_predicate() {
                    Ok(Self(attr))
                } else {
                    Err(InvalidTypeCastError)
                }
            }
        }
    };
}

/// Represents an ad-hoc kind/value attribute, of which both components are strings
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct StringAttribute(AttributeBase);
impl_attr_traits!(StringAttribute, string);
impl Attribute for StringAttribute {
    #[inline(always)]
    fn is_string_attribute(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_enum_attribute(&self) -> bool {
        false
    }

    fn base(&self) -> AttributeBase {
        self.0
    }

    fn remove(&self, fun: Function, index: AttributePlace) {
        extern "C" {
            fn LLVMRemoveStringAttributeAtIndex(
                fun: Function,
                index: u32,
                kind: *const u8,
                kind_len: u32,
            );
        }
        let kind = self.kind();
        unsafe {
            LLVMRemoveStringAttributeAtIndex(
                fun,
                index.into(),
                kind.data,
                kind.len.try_into().unwrap(),
            )
        }
    }
}
impl StringAttribute {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn new<K, V>(context: Context, kind: K, value: V) -> Self
    where
        K: Into<StringRef>,
        V: Into<StringRef>,
    {
        extern "C" {
            fn LLVMCreateStringAttribute(
                context: Context,
                k: *const u8,
                klen: u32,
                v: *const u8,
                vlen: u32,
            ) -> AttributeBase;
        }
        let kind = kind.into();
        let value = value.into();
        let attr = unsafe {
            LLVMCreateStringAttribute(
                context,
                kind.data,
                kind.len.try_into().unwrap(),
                value.data,
                value.len.try_into().unwrap(),
            )
        };
        debug_assert!(attr.is_string_attribute());
        Self(attr)
    }

    pub fn kind(&self) -> StringRef {
        extern "C" {
            fn LLVMGetStringAttributeKind(attr: StringAttribute, len: *mut u32) -> *const u8;
        }
        let mut len = MaybeUninit::uninit();
        unsafe {
            let ptr = LLVMGetStringAttributeKind(*self, len.as_mut_ptr());
            assert!(!ptr.is_null());
            StringRef::from_raw_parts(ptr, len.assume_init() as usize)
        }
    }

    pub fn value(&self) -> StringRef {
        extern "C" {
            fn LLVMGetStringAttributeValue(attr: StringAttribute, len: *mut u32) -> *const u8;
        }
        let mut len = MaybeUninit::uninit();
        unsafe {
            let ptr = LLVMGetStringAttributeValue(*self, len.as_mut_ptr());
            assert!(!ptr.is_null());
            StringRef::from_raw_parts(ptr, len.assume_init() as usize)
        }
    }
}

/// Represents an known attribute defined in LLVM's attribute kind enumeration
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct EnumAttribute(AttributeBase);
impl_attr_traits!(EnumAttribute, enum);
impl Attribute for EnumAttribute {
    #[inline(always)]
    fn is_string_attribute(&self) -> bool {
        false
    }

    #[inline(always)]
    fn is_enum_attribute(&self) -> bool {
        true
    }

    fn base(&self) -> AttributeBase {
        self.0
    }

    fn remove(&self, fun: Function, index: AttributePlace) {
        extern "C" {
            fn LLVMRemoveEnumAttributeAtIndex(fun: Function, index: u32, kind: AttributeKind);
        }
        unsafe { LLVMRemoveEnumAttributeAtIndex(fun, index.into(), self.kind()) }
    }
}

impl EnumAttribute {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn get<S: Into<StringRef>>(name: S) -> Option<AttributeKind> {
        extern "C" {
            fn LLVMGetEnumAttributeKindForName(name: *const u8, len: usize) -> AttributeKind;
        }
        let name = name.into();
        let kind = unsafe { LLVMGetEnumAttributeKindForName(name.data, name.len) };
        if kind == AttributeKind::Invalid {
            None
        } else {
            Some(kind)
        }
    }

    /// Creates a new instance of the attribute with the given kind
    ///
    /// The value is optional for most enum attributes, but a handful accept or require a value:
    ///
    /// * alignment (indicates that the pointer or vector of pointers has the specified alignment)
    /// * allocsize (indicates the annotated function will always return at least the given number of bytes or null)
    /// * dereferenceable (indicates that the parameter/return pointer is dereferenceable with the given number of bytes)
    /// * dereferenceable_or_null (indicates the parameter/return pointer is either null or dereferenceable to N bytes)
    /// * alignstack (specifies the desired alignment, power of two)
    /// * uwtable (if value is given, specifies what kind of unwind tables to generate, 1 = sync/normal, 2 = async/instruction precise)
    /// * vscale_range (the minimum vscale value for the given function, must be greater than 0)
    pub fn new(context: Context, kind: AttributeKind, value: Option<u64>) -> Self {
        extern "C" {
            fn LLVMCreateEnumAttribute(
                context: Context,
                kind: AttributeKind,
                value: u64,
            ) -> AttributeBase;
        }
        let value = match kind {
            // If no value is provided for uwtable, the default is 2 (async/instruction precise)
            AttributeKind::UWTable if value.is_none() => 2,
            // If no value is provided for these attributes, we panic
            AttributeKind::Alignment
            | AttributeKind::AllocSize
            | AttributeKind::StackAlignment
            | AttributeKind::VScaleRange
                if value.is_none() =>
            {
                panic!("value is required for {:?} attribute", kind)
            }
            // The default for all others is 0
            _ => value.unwrap_or_default(),
        };
        let attr = unsafe { LLVMCreateEnumAttribute(context, kind, value) };
        debug_assert!(attr.is_enum_attribute());
        Self(attr)
    }

    pub fn kind(&self) -> AttributeKind {
        extern "C" {
            fn LLVMGetEnumAttributeKind(attr: EnumAttribute) -> AttributeKind;
        }
        unsafe { LLVMGetEnumAttributeKind(*self) }
    }

    pub fn value(&self) -> u64 {
        extern "C" {
            fn LLVMGetEnumAttributeValue(attr: AttributeBase) -> u64;
        }
        unsafe { LLVMGetEnumAttributeValue(self.0) }
    }
}

/// Represents an known attribute defined in LLVM's attribute kind enumeration that has a type parameter
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TypeAttribute(AttributeBase);
impl_attr_traits!(TypeAttribute, type);
impl Attribute for TypeAttribute {
    #[inline(always)]
    fn is_type_attribute(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_string_attribute(&self) -> bool {
        false
    }

    fn base(&self) -> AttributeBase {
        self.0
    }

    fn remove(&self, fun: Function, index: AttributePlace) {
        extern "C" {
            fn LLVMRemoveEnumAttributeAtIndex(fun: Function, index: u32, kind: AttributeKind);
        }
        unsafe { LLVMRemoveEnumAttributeAtIndex(fun, index.into(), self.kind()) }
    }
}
impl TypeAttribute {
    pub fn get<T: Type>(ty: T, kind: AttributeKind) -> Self {
        extern "C" {
            fn LLVMCreateTypeAttribute(
                context: Context,
                kind: AttributeKind,
                ty: TypeBase,
            ) -> AttributeBase;
        }
        let context = ty.context();
        let attr = unsafe { LLVMCreateTypeAttribute(context, kind, ty.base()) };
        debug_assert!(attr.is_type_attribute());
        Self(attr)
    }

    pub fn kind(&self) -> AttributeKind {
        extern "C" {
            fn LLVMGetEnumAttributeKind(attr: AttributeBase) -> AttributeKind;
        }
        unsafe { LLVMGetEnumAttributeKind(self.0) }
    }

    pub fn value(&self) -> TypeBase {
        extern "C" {
            fn LLVMGetTypeAttributeValue(attr: TypeAttribute) -> TypeBase;
        }
        unsafe { LLVMGetTypeAttributeValue(*self) }
    }
}

/// Represents the place of a function on which an attribute is placed
#[derive(Copy, Clone)]
pub enum AttributePlace {
    ReturnValue,
    Argument(u32),
    Function,
}
impl Into<u32> for AttributePlace {
    fn into(self) -> u32 {
        match self {
            Self::ReturnValue => 0,
            Self::Argument(n) => 1 + n,
            Self::Function => u32::MAX,
        }
    }
}
impl From<u32> for AttributePlace {
    fn from(n: u32) -> Self {
        match n {
            0 => Self::ReturnValue,
            u32::MAX => Self::Function,
            n => Self::Argument(n - 1),
        }
    }
}

/// In LLVM these values are generated by TableGen, ideally we'd also generate this enum,
/// but for now we're hardcoding it and keeping it up to date by hand
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttributeKind {
    Invalid = 0,
    // Enum attributes
    AllocAlign = 1,
    AlwaysInline = 2,
    ArgMemOnly = 3,
    Builtin = 4,
    Cold = 5,
    Convergent = 6,
    DisableSanitizerInstrumentation = 7,
    Hot = 8,
    ImmArg = 9,
    InReg = 10,
    InaccessibleMemOnly = 11,
    InaccessibleMemOrArgMemOnly = 12,
    InlineHint = 13,
    JumpTable = 14,
    MinSize = 15,
    MustProgress = 16,
    Naked = 17,
    Nest = 18,
    NoAlias = 19,
    NoBuiltin = 20,
    NoCallback = 21,
    NoCapture = 22,
    NoCfCheck = 23,
    NoDuplicate = 24,
    NoFree = 25,
    NoImplicitFloat = 26,
    NoInline = 27,
    NoMerge = 28,
    NoProfile = 29,
    NoRecurse = 30,
    NoRedZone = 31,
    NoReturn = 32,
    NoSanitizeBounds = 33,
    NoSanitizeCoverage = 34,
    NoSync = 35,
    NoUndef = 36,
    NoUnwind = 37,
    NonLazyBind = 38,
    NonNull = 39,
    NullPointerIsValid = 40,
    OptForFuzzing = 41,
    OptimizeForSize = 42,
    OptimizeNone = 43,
    ReadNone = 44,
    ReadOnly = 45,
    Returned = 46,
    ReturnsTwice = 47,
    SExt = 48,
    SafeStack = 49,
    SanitizeAddress = 50,
    SanitizeHWAddress = 51,
    SanitizeMemTag = 52,
    SanitizeMemory = 53,
    SanitizeThread = 54,
    ShadowCallStack = 55,
    Speculatable = 56,
    SpeculativeLoadHardening = 57,
    StackProtect = 58,
    StackProtectReq = 59,
    StackProtectStrong = 60,
    StrictFP = 61,
    SwiftAsync = 62,
    SwiftError = 63,
    SwiftSelf = 64,
    WillReturn = 65,
    WriteOnly = 66,
    ZExt = 67,
    // Type attributes
    ByRef = 68,
    ByVal = 69,
    ElementType = 70,
    InAlloca = 71,
    Preallocated = 72,
    StructRet = 73,
    // Integer attributes
    Alignment = 74,
    AllocSize = 75,
    Dereferenceable = 76,
    DereferenceableOrNull = 77,
    StackAlignment = 78,
    UWTable = 79,
    VScaleRange = 80,
}
