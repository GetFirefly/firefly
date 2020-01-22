#ifndef EIR_TYPES_TD
#define EIR_TYPES_TD

//===----------------------------------------------------------------------===//
// EIR Types
//===----------------------------------------------------------------------===//

// Atoms
def eir_AtomType : Type<CPred<"$_self.isa<eir::AtomType>()">,
    "EIR atom type">;
def eir_BooleanType : Type<CPred<"$_self.isa<eir::BooleanType>()">,
    "EIR boolean type">;

// Type constraint for boolean-like types
def AnyBooleanLike : TypeConstraint<Or<[
    BoolLike.predicate,
    eir_BooleanType.predicate]>, "any boolean">;

// Integers

def eir_FixnumType : Type<CPred<"$_self.isa<eir::FixnumType>()">,
    "EIR fixed-width integer type">;
def eir_BigIntType : Type<CPred<"$_self.isa<eir::BigIntType>()">,
    "EIR arbitrary-width integer type">;
def AnyIntegerLike : TypeConstraint<Or<[
    IntegerLike.predicate,
    eir_FixnumType.predicate,
    eir_BigIntType.predicate]>, "any integer">;
def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

// Floats

def eir_FloatType : Type<CPred<"$_self.isa<eir::FloatType>()">,
    "EIR immediate float type">;
def eir_PackedFloatType : Type<CPred<"$_self.isa<eir::PackedFloatType>()">,
    "EIR packed float type">;
def AnyFloatLike : TypeConstraint<Or<[
    FloatLike.predicate,
    eir_FloatType.predicate,
    eir_PackedFloatType.predicate]>, "any float">;
def AnyFloatType : Type<AnyFloatLike.predicate, "any float">;

def AnyNumberLike : TypeConstraint<Or<[
    AnyIntegerLike.predicate,
    AnyFloatLike.predicate]>, "any float or integer">;

def NumberType : Type<AnyNumberLike.predicate, "any integer or float">;

// Cons/Lists

def eir_NilType : Type<CPred<"$_self.isa<eir::NilType>()">,
    "EIR nil type">;
def eir_ConsType : Type<CPred<"$_self.isa<eir::ConsType>()">,
    "EIR cons cell type">;
def AnyListLike : TypeConstraint<Or<[
    eir_NilType.predicate,
    eir_ConsType.predicate]>, "any list-like type">;
def ListType : Type<AnyListLike.predicate, "any list">;

// Tuples

def eir_TupleType : Type<CPred<"$_self.isa<eir::TupleType>()">,
    "EIR tuple type">;

// Maps

def eir_MapType : Type<CPred<"$_self.isa<eir::MapType>()">,
    "EIR map type">;

// Closures

def eir_ClosureType : Type<CPred<"$_self.isa<eir::ClosureType>()">,
    "EIR closure type">;

// Binaries

def eir_BinaryType : Type<CPred<"$_self.isa<eir::BinaryType>()">,
    "EIR binary literal type">;
def eir_HeapBinType : Type<CPred<"$_self.isa<eir::HeapBinType>()">,
    "EIR heap-allocated binary type">;
//def eir_ProcBinType : Type<CPred<"$_self.isa<eir::ProcBinType>()">,
//    "EIR reference-counted binary type">;
//def eir_SubBinaryType : Type<CPred<"$_self.isa<eir::SubBinaryType>()">,
//    "EIR sub-binary slice">;
//def eir_MatchContextType : Type<CPred<"$_self.isa<eir::MatchContextType>()">,
//    "EIR binary/bitstring match context">;

def BinaryLike : TypeConstraint<Or<[
    eir_BinaryType.predicate,
    eir_HeapBinType.predicate,
    //eir_ProcBinType.predicate,
    //eir_SubBinaryType.predicate,
    //eir_MatchContextType.predicate,
]>, "any binary type">;

def AnyBinaryType : Type<BinaryLike.predicate, "any binary">;

// Boxed Values
//   !eir.box = A reference to a term on the process heap
def eir_BoxType : Type<CPred<"$_self.isa<eir::BoxType>()">, "box type">;

// References/Pointers
//   !eir.ref = A mutable memory reference, allocated on
//              the stack generally, but could also be on
//              the global heap in theory.
def eir_RefType : Type<CPred<"$_self.isa<eir::RefType>()">,
    "reference type">;

def AnyRefLike : TypeConstraint<Or<[
    eir_RefType.predicate,
    eir_BoxType.predicate]>, "any reference/pointer-like type">;

// Immediate Values

def AnyImmediateLike : TypeConstraint<Or<[
    eir_BoxType.predicate,
    eir_AtomType.predicate,
    eir_BooleanType.predicate,
    eir_FixnumType.predicate,
    eir_FloatType.predicate,
    eir_NilType.predicate]>, "any immediate/word-sized type">;

// Base type that represents any valid term type (i.e. defined by the EIR dialect)
def eir_TermType : Type<CPred<"$_self.isa<eir::TermType>()">,
    "EIR term type">;

def AnyTermLike : TypeConstraint<Or<[
    AnyImmediateLike.predicate,
    eir_BoxType.predicate]>, "any valid term type">;

def AnyTermType : Type<AnyTermLike.predicate, "any valid term">;

#endif
