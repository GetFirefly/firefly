//===- JSONBackend.cpp - Generate a JSON dump of all records. -*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This TableGen back end generates a machine-readable representation
// of all the classes and records defined by the input, in JSON format.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/JSON.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#define DEBUG_TYPE "json-emitter"

using namespace mlir;
using namespace mlir::tblgen;
using ::llvm::DagInit;
using ::llvm::DefInit;
using ::llvm::Init;
using ::llvm::Record;
using ::llvm::RecordVal;

namespace json = llvm::json;

static llvm::cl::OptionCategory clOpJsonDumpCat("Options for -dump-json");

namespace {

/// Utility iterator used for filtering records for a specific dialect.
using DialectFilterIterator =
    llvm::filter_iterator<ArrayRef<llvm::Record *>::iterator,
                          std::function<bool(const llvm::Record *)>>;

/// Given a set of records for a T, filter the ones that correspond to
/// the given dialect.
template <typename T>
static iterator_range<DialectFilterIterator> filterForDialect(
    ArrayRef<llvm::Record *> records, Dialect &dialect) {
    auto filterFn = [&](const llvm::Record *record) {
        const RecordVal *rec = record->getValue("dialect");
        if (!rec || !rec->getValue()) return false;
        return T(record).getDialect() == dialect;
    };
    return {DialectFilterIterator(records.begin(), records.end(), filterFn),
            DialectFilterIterator(records.end(), records.end(), filterFn)};
}

static iterator_range<DialectFilterIterator> filterOpsForDialect(
    ArrayRef<llvm::Record *> records, const Record *dialect) {
    auto filterFn = [=](const llvm::Record *record) {
        const RecordVal *rec = record->getValue("opDialect");
        if (!rec || !rec->getValue()) return false;
        auto *value = rec->getValue();
        return value->getAsString() == dialect->getName();
    };
    return {DialectFilterIterator(records.begin(), records.end(), filterFn),
            DialectFilterIterator(records.end(), records.end(), filterFn)};
}

class JsonEmitter {
   private:
    const llvm::RecordKeeper &recordKeeper;

    json::Value dialectAttrsToJsonValue(Dialect &,
                                        iterator_range<DialectFilterIterator>,
                                        iterator_range<DialectFilterIterator>);
    json::Value dialectEnumsToJsonValue(Dialect &, ArrayRef<Record *>);
    json::Value dialectTypesToJsonValue(Dialect &,
                                        iterator_range<DialectFilterIterator>,
                                        iterator_range<DialectFilterIterator>);
    json::Value dialectOpsToJsonValue(Dialect &,
                                      iterator_range<DialectFilterIterator>);
    json::Object opToJsonValue(Operator &, const Record *);
    json::Object attrOrTypeDefToJsonValue(const tblgen::AttrOrTypeDef &,
                                          const Record *);
    json::Object typeToJsonValue(const tblgen::Type &, const Record *);
    json::Object typeToJsonValue(const tblgen::TypeConstraint &,
                                 const Record *);
    json::Object attrToJsonValue(const tblgen::Attribute &, const Record *);
    json::Object attrToJsonValue(json::Object, const tblgen::ConstantAttr &);
    json::Object attrToJsonValue(json::Object, const tblgen::EnumAttr &);
    json::Object attrToJsonValue(json::Object, const tblgen::EnumAttrCase &);
    json::Object attrToJsonValue(json::Object, const tblgen::StructAttr &,
                                 const Record *);
    json::Object attrToJsonValue(json::Object, const tblgen::StructFieldAttr &,
                                 const Record *);
    json::Object enumToJsonValue(const tblgen::EnumAttr &, const Record *);

    std::string printAnonymousClassType(Record *);

   public:
    JsonEmitter(const llvm::RecordKeeper &rs) : recordKeeper(rs) {}

    bool run(llvm::raw_ostream &os);
};
}  // end anonymous namespace

json::Object JsonEmitter::typeToJsonValue(const tblgen::Type &ty,
                                          const Record *def) {
    json::Object obj;
    obj["class"] = StringRef("Type");
    auto *dialect = def->getValue("dialect");
    if (dialect) {
        auto *dialectVal = dialect->getValue();
        if (dialectVal) {
            obj["dialect"] = ty.getDialect().getName();
        }
    }
    auto *typeDesc = def->getValue("typeDescription");
    if (typeDesc) {
        obj["description"] = ty.getDescription();
    }

    obj["isOptional"] = ty.isOptional();
    obj["isVariadic"] = ty.isVariadic();
    obj["cppClass"] = ty.getCPPClassName();

    const Record *baseType = def;
    if (ty.isVariableLength()) {
        baseType = def->getValueAsDef("baseType");
    }
    obj["name"] = baseType->getName();

    return obj;
}

json::Object JsonEmitter::typeToJsonValue(const tblgen::TypeConstraint &ty,
                                          const Record *def) {
    json::Object obj;
    obj["class"] = StringRef("Type");
    obj["isOptional"] = ty.isOptional();
    obj["isVariadic"] = ty.isVariadic();
    obj["cppClass"] = ty.getCPPClassName();

    const Record *baseType = def;
    if (ty.isVariableLength()) {
        baseType = def->getValueAsDef("baseType");
    }
    obj["name"] = baseType->getName();

    return obj;
}

json::Object JsonEmitter::attrOrTypeDefToJsonValue(
    const tblgen::AttrOrTypeDef &attrOrType, const Record *def) {
    json::Object obj;
    if (isa<AttrDef>(attrOrType)) {
        obj["class"] = StringRef("AttrDef");
    } else {
        obj["class"] = StringRef("TypeDef");
    }
    obj["dialect"] = attrOrType.getDialect().getName();
    obj["name"] = attrOrType.getName();
    if (attrOrType.hasDescription()) {
        obj["description"] = attrOrType.getDescription();
    }
    if (attrOrType.hasSummary()) {
        obj["summary"] = attrOrType.getSummary();
    }
    obj["cppClass"] = attrOrType.getCppClassName();
    obj["baseClass"] = attrOrType.getCppBaseClassName();
    obj["storageClass"] = attrOrType.getStorageClassName();
    obj["storageNamespace"] = attrOrType.getStorageNamespace();
    obj["skipDefaultBuilders"] = attrOrType.skipDefaultBuilders();

    if (Optional<StringRef> mnemonic = attrOrType.getMnemonic()) {
        obj["mnemonic"] = *mnemonic;
    }

    // Builders
    json::Array builders;
    for (const AttrOrTypeBuilder &builder : attrOrType.getBuilders()) {
        json::Object bldr;
        bldr["hasInferredContextParameter"] =
            builder.hasInferredContextParameter();

        json::Array buildParams;
        for (const AttrOrTypeBuilder::Parameter &param :
             builder.getParameters()) {
            json::Object p;
            p["cppType"] = param.getCppType();
            if (Optional<StringRef> name = param.getName()) {
                p["name"] = *name;
            }
            if (Optional<StringRef> defaultParamValue =
                    param.getDefaultValue()) {
                p["defaultValue"] = *defaultParamValue;
            }
            buildParams.push_back(std::move(p));
        }
        bldr["parameters"] = std::move(buildParams);
        builders.push_back(std::move(bldr));
    }
    obj["builders"] = std::move(builders);

    // Parameters
    json::Array parameters;
    SmallVector<AttrOrTypeParameter, 4> params;
    attrOrType.getParameters(params);
    for (AttrOrTypeParameter &param : params) {
        json::Object p;
        p["name"] = param.getName();
        p["cppType"] = param.getCppType();
        if (Optional<StringRef> summary = param.getSummary()) {
            p["summary"] = *summary;
        }
        parameters.push_back(std::move(p));
    }
    obj["parameters"] = std::move(parameters);

    return obj;
}

json::Object JsonEmitter::attrToJsonValue(const tblgen::Attribute &attr,
                                          const Record *def) {
    json::Object obj;
    obj["class"] = StringRef("Attr");
    obj["name"] = attr.getAttrDefName();

    Attribute baseAttr = attr.getBaseAttr();
    obj["baseAttr"] = baseAttr.getAttrDefName();

    obj["isOptional"] = attr.isOptional();
    obj["storageType"] = attr.getStorageType();
    obj["returnType"] = attr.getReturnType();
    if (auto *valTy = dyn_cast<DefInit>(def->getValueInit("valueType"))) {
        const Record *tyDef = valTy->getDef();
        obj["valueType"] = typeToJsonValue(Type(tyDef), tyDef);
    }

    if (attr.hasDefaultValue()) {
        obj["defaultValue"] = attr.getDefaultValue();
    }

    if (attr.isEnumAttr()) {
        return attrToJsonValue(std::move(obj), cast<tblgen::EnumAttr>(attr));
    } else {
        return obj;
    }
}

json::Object JsonEmitter::attrToJsonValue(json::Object obj,
                                          const tblgen::ConstantAttr &attr) {
    obj["class"] = StringRef("ConstantAttr");
    obj["constantValue"] = attr.getConstantValue();
    return obj;
}

json::Object JsonEmitter::attrToJsonValue(json::Object obj,
                                          const tblgen::EnumAttr &attr) {
    obj["class"] = StringRef("EnumAttr");
    obj["isBitEnum"] = attr.isBitEnum();
    obj["className"] = attr.getEnumClassName();
    obj["namespace"] = attr.getCppNamespace();
    obj["underlyingType"] = attr.getUnderlyingType();
    json::Array cases;
    auto enumCases = attr.getAllCases();
    for (const tblgen::EnumAttrCase &ec : enumCases) {
        json::Object caseObj;
        cases.push_back(attrToJsonValue(std::move(caseObj), ec));
    }
    obj["cases"] = std::move(cases);
    return obj;
}

json::Object JsonEmitter::attrToJsonValue(json::Object obj,
                                          const tblgen::EnumAttrCase &attr) {
    obj["isStr"] = attr.isStrCase();
    obj["symbol"] = attr.getSymbol();
    obj["text"] = attr.getStr();
    obj["value"] = attr.getValue();
    return obj;
}

json::Object JsonEmitter::attrToJsonValue(json::Object obj,
                                          const tblgen::StructAttr &attr,
                                          const Record *def) {
    obj["class"] = StringRef("StructAttr");
    obj["className"] = attr.getStructClassName();
    obj["namespace"] = attr.getCppNamespace();
    json::Array fields;
    auto structFields = attr.getAllFields();
    auto *fieldDefs = def->getValueAsListInit("fields");
    for (auto it : llvm::enumerate(structFields)) {
        json::Object fieldObj;
        auto &f = it.value();
        auto *def = fieldDefs->getElementAsRecord(it.index());
        fields.push_back(attrToJsonValue(std::move(fieldObj), f, def));
    }
    obj["fields"] = std::move(fields);
    return obj;
}

json::Object JsonEmitter::attrToJsonValue(json::Object obj,
                                          const tblgen::StructFieldAttr &attr,
                                          const Record *def) {
    auto *init = def->getValueInit("type");
    auto *fieldType = cast<DefInit>(*init).getDef();
    obj["name"] = attr.getName();
    obj["type"] = attrToJsonValue(Attribute(fieldType), fieldType);
    return obj;
}

std::string JsonEmitter::printAnonymousClassType(Record *def) {
    auto *attrClass = recordKeeper.getClass("Attr");
    auto *typeClass = recordKeeper.getClass("Type");
    auto *opVarClass = recordKeeper.getClass("OpVariable");
    auto *typeConstraintClass = recordKeeper.getClass("TypeConstraint");
    auto *variadicClass = recordKeeper.getClass("Variadic");
    auto *optionalClass = recordKeeper.getClass("Optional");
    auto *confinedClass = recordKeeper.getClass("Confined");
    auto *anyTypeOfClass = recordKeeper.getClass("AnyTypeOf");
    auto *predOrClass = recordKeeper.getClass("Or");
    auto *predAndClass = recordKeeper.getClass("And");

    auto defName = def->getName();

    if (def->isSubClassOf(attrClass)) return std::string("attribute");
    if (!def->isAnonymous()) return defName.str();

    llvm::errs() << "def " << defName << "\n";
    llvm::errs() << *def;

    auto anonName = def->getName();
    auto *anonDef = recordKeeper.getDef(anonName);
    auto isVariadic = anonDef->hasDirectSuperClass(variadicClass);
    auto isOptional = anonDef->hasDirectSuperClass(optionalClass);

    SmallVector<Record *, 1> classes;
    anonDef->getDirectSuperClasses(classes);

    std::string result;

    if (isVariadic || isOptional) {
        auto *baseType = anonDef->getValueAsDef("baseType");
        auto *baseTypeDef = recordKeeper.getDef(baseType->getName());
        auto baseTypeName = printAnonymousClassType(baseTypeDef);

        if (isVariadic) {
            return std::string(std::string("ArrayRef<") + baseTypeName +
                               std::string(">"));
        } else {
            return std::string(std::string("Optional<") + baseTypeName +
                               std::string(">"));
        }
    } else {
        result += anonName;
    }

    // llvm::errs() << *anonDef;

    if (auto predVal = anonDef->getValue("predicate")) {
        auto *classPred = dyn_cast<DefInit>(predVal->getValue());
        auto classPredName = classPred->getDef()->getName();
        auto *pred = recordKeeper.getDef(classPredName);
        bool isKnownPredKind = false;
        result += classPredName;
        result += std::string("<");

        if (pred->hasDirectSuperClass(predOrClass)) {
            // Or<children..>
            isKnownPredKind = true;
            result += std::string("Or<");
        } else if (pred->hasDirectSuperClass(predAndClass)) {
            // And<[children..]>
            isKnownPredKind = true;
            result += std::string("And<");
        } else {
            // Probably a CPred, but ignore it
            // llvm::errs() << *pred;
        }

        if (isKnownPredKind) {
            auto children = pred->getValueAsListOfDefs("children");
            auto numChildren = children.size();
            for (auto it : llvm::enumerate(children)) {
                auto *child = it.value();
                auto index = it.index();
                auto *childDef = recordKeeper.getDef(child->getName());
                auto childTy = printAnonymousClassType(childDef);
                if (numChildren > index + 1) {
                    result = result + childTy + std::string(",");
                } else {
                    result = result + childTy;
                }
            }
            result += std::string(">");
        } else {
            result += std::string("T");
        }
        result += std::string(">");
    } else {
        result += std::string("unknown");
    }

    return result;
}

json::Object JsonEmitter::opToJsonValue(Operator &op, const Record *def) {
    auto *attrClass = recordKeeper.getClass("Attr");
    auto *opVarClass = recordKeeper.getClass("OpVariable");
    auto *typeConstraintClass = recordKeeper.getClass("TypeConstraint");
    auto *variadicClass = recordKeeper.getClass("Variadic");
    auto *optionalClass = recordKeeper.getClass("Optional");
    auto *confinedClass = recordKeeper.getClass("Confined");
    auto *anyTypeOfClass = recordKeeper.getClass("AnyTypeOf");
    auto *predOrClass = recordKeeper.getClass("Or");
    auto *predAndClass = recordKeeper.getClass("And");

    json::Object obj;
    obj["name"] = op.getCppClassName();
    obj["dialect"] = op.getDialectName();
    obj["mnemonic"] = op.getOperationName();
    obj["cppClass"] = op.getQualCppClassName();
    obj["isVariadic"] = op.isVariadic();
    obj["skipDefaultBuilders"] = op.skipDefaultBuilders();

    // Need to build up the set of Records for attributes
    DagInit *argumentsDag = def->getValueAsDag("arguments");
    DagInit *resultsDag = def->getValueAsDag("results");
    unsigned numArgs = argumentsDag->getNumArgs();
    unsigned numResults = resultsDag->getNumArgs();

    // Operands/Attributes
    json::Array operands;
    json::Object attributes;

    for (unsigned i = 0; i != numArgs; ++i) {
        auto *arg = argumentsDag->getArg(i);
        auto *argDefInit = dyn_cast<DefInit>(arg);
        assert(argDefInit != nullptr && "expected DefInit");

        Record *argDef = argDefInit->getDef();

        llvm::errs() << "arg type is: " << printAnonymousClassType(argDef)
                     << "\n";

        if (argDef->isSubClassOf(opVarClass))
            argDef = argDef->getValueAsDef("constraint");

        if (argDef->isSubClassOf(typeConstraintClass)) {
            auto argDefObj = typeToJsonValue(TypeConstraint(argDef), argDef);
            argDefObj["name"] = argumentsDag->getArgNameStr(i);
            argDefObj["index"] = i;
            operands.push_back(std::move(argDefObj));
        } else if (argDef->isSubClassOf(attrClass)) {
            auto argDefObj = attrToJsonValue(Attribute(argDef), argDef);
            auto givenName = argumentsDag->getArgNameStr(i);
            argDefObj["name"] = givenName;
            argDefObj["index"] = i;
            attributes[givenName] = attrToJsonValue(Attribute(argDef), argDef);
            operands.push_back(std::move(argDefObj));
        }
    }
    obj["operands"] = std::move(operands);
    obj["attributes"] = std::move(attributes);

    /// Results
    json::Array results;
    for (unsigned i = 0; i < numResults; ++i) {
        auto name = resultsDag->getArgNameStr(i);
        Init *arg = resultsDag->getArg(i);
        auto *resultInit = dyn_cast<DefInit>(arg);
        assert(resultInit && "expected DefInit");

        auto *resultDef = resultInit->getDef();
        if (resultDef->isSubClassOf(opVarClass))
            resultDef = resultDef->getValueAsDef("constraint");

        auto resultObj = typeToJsonValue(TypeConstraint(resultDef), resultDef);
        resultObj["name"] = name;
        resultObj["index"] = i;
        results.push_back(std::move(resultObj));
    }
    obj["results"] = std::move(results);

    /// Successors
    json::Array successors;
    for (const NamedSuccessor &succ : op.getSuccessors()) {
        json::Object succObj;
        succObj["name"] = succ.name;
        succObj["isVariadic"] = succ.isVariadic();
        successors.push_back(std::move(succObj));
    }
    obj["successors"] = std::move(successors);

    /// Regions
    json::Array regions;
    for (const NamedRegion &region : op.getRegions()) {
        json::Object regionObj;
        regionObj["name"] = region.name;
        regionObj["isVariadic"] = region.isVariadic();
        regions.push_back(std::move(regionObj));
    }
    obj["regions"] = std::move(regions);

    return obj;
}

json::Value JsonEmitter::dialectAttrsToJsonValue(
    Dialect &dialect, iterator_range<DialectFilterIterator> dialectAttrDecls,
    iterator_range<DialectFilterIterator> dialectAttrDefs) {
    json::Array attrs;

    for (const Record *def : dialectAttrDecls) {
        Attribute attr(def);
        attrs.push_back(attrToJsonValue(attr, def));
    }

    for (const Record *def : dialectAttrDefs) {
        AttrOrTypeDef attr(def);
        attrs.push_back(attrOrTypeDefToJsonValue(attr, def));
    }

    return attrs;
}

json::Value JsonEmitter::dialectEnumsToJsonValue(
    Dialect &dialect, ArrayRef<Record *> dialectEnums) {
    json::Array enums;

    for (const Record *def : dialectEnums) {
        EnumAttr attr(def);
        enums.push_back(attrToJsonValue(attr, def));
    }

    return enums;
}

json::Value JsonEmitter::dialectTypesToJsonValue(
    Dialect &dialect, iterator_range<DialectFilterIterator> dialectTypeDecls,
    iterator_range<DialectFilterIterator> dialectTypeDefs) {
    json::Array types;

    for (const Record *def : dialectTypeDecls) {
        Type ty(def);
        types.push_back(typeToJsonValue(ty, def));
    }

    for (const Record *def : dialectTypeDefs) {
        AttrOrTypeDef ty(def);
        types.push_back(attrOrTypeDefToJsonValue(ty, def));
    }

    return types;
}

json::Value JsonEmitter::dialectOpsToJsonValue(
    Dialect &dialect, iterator_range<DialectFilterIterator> dialectOps) {
    json::Array ops;

    for (const Record *def : dialectOps) {
        Operator op(def);
        ops.push_back(opToJsonValue(op, def));
    }

    return ops;
}

bool JsonEmitter::run(llvm::raw_ostream &os) {
    json::Array dialects;

    auto attrDecls = recordKeeper.getAllDerivedDefinitions("DialectAttr");
    auto attrDefs = recordKeeper.getAllDerivedDefinitions("AttrDef");
    auto enumDefs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
    auto typeDecls = recordKeeper.getAllDerivedDefinitions("DialectType");
    auto typeDefs = recordKeeper.getAllDerivedDefinitions("TypeDef");
    auto dialectDefs = recordKeeper.getAllDerivedDefinitions("Dialect");
    auto opDefs = recordKeeper.getAllDerivedDefinitions("Op");

    for (auto dialectDef : dialectDefs) {
        json::Object dialectObj;

        Dialect dialect(dialectDef);
        dialectObj["name"] = dialect.getName();
        dialectObj["namespace"] = dialect.getCppNamespace();
        dialectObj["className"] = dialect.getCppClassName();
        dialectObj["summary"] = dialect.getSummary();
        dialectObj["description"] = dialect.getDescription();
        json::Array dialectDependencies;
        for (auto dep : dialect.getDependentDialects()) {
            dialectDependencies.push_back(dep);
        }
        dialectObj["dependencies"] = std::move(dialectDependencies);
        dialectObj["attributes"] = dialectAttrsToJsonValue(
            dialect, filterForDialect<Attribute>(attrDecls, dialect),
            filterForDialect<AttrDef>(attrDefs, dialect));
        dialectObj["enums"] = dialectEnumsToJsonValue(dialect, enumDefs);
        dialectObj["types"] = dialectTypesToJsonValue(
            dialect, filterForDialect<Type>(typeDecls, dialect),
            filterForDialect<TypeDef>(typeDefs, dialect));
        dialectObj["operations"] = dialectOpsToJsonValue(
            dialect, filterOpsForDialect(opDefs, dialectDef));
        dialects.push_back(std::move(dialectObj));
    }

    os << llvm::formatv("{0:2}", json::Value(std::move(dialects))) << "\n";
    return false;
}

static bool emitJson(const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
    return JsonEmitter(records).run(os);
}

static GenRegistration dumpJson(
    "dump-json",
    "Dump JSON representation of given MLIR-flavored TableGen file", &emitJson);
