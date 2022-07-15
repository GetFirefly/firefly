extern crate cmake;

use std::env;
use std::path::PathBuf;

const ENV_LLVM_CORE_INCLUDE: &'static str = "DEP_LUMEN_LLVM_CORE_INCLUDE";
const ENV_LLVM_PREFIX: &'static str = "DEP_LUMEN_LLVM_CORE_PREFIX";
const ENV_LLVM_LINK_STATIC: &'static str = "DEP_LUMEN_LLVM_CORE_LINK_STATIC";
const ENV_LLVM_LINK_LLVM_DYLIB: &'static str = "DEP_LUMEN_LLVM_CORE_LINK_LLVM_DYLIB";
const ENV_LLVM_LTO: &'static str = "DEP_LUMEN_LLVM_CORE_LTO";
const ENV_LLVM_USE_SANITIZER: &'static str = "LLVM_USE_SANITIZER";

fn main() {
    let lumen_llvm_include_dir = env::var(ENV_LLVM_CORE_INCLUDE).unwrap();
    let llvm_prefix = PathBuf::from(env::var(ENV_LLVM_PREFIX).unwrap());
    let mlir_dir = llvm_prefix.join("lib/cmake/mlir");
    let llvm_dir = llvm_prefix.join("lib/cmake/llvm");
    let link_lto = env::var(ENV_LLVM_LTO).unwrap();

    println!("cargo:rerun-if-changed=c_src");
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_CORE_INCLUDE);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_LINK_STATIC);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_LINK_LLVM_DYLIB);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_LTO);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_USE_SANITIZER);

    // Build and link our MLIR dialects + extensions
    let mut config = cmake::Config::new("c_src");

    if let Ok(_) = which::which("ninja") {
        config.generator("Ninja");
    } else {
        warn(
            "Unable to locate Ninja, your CMake builds may take unncessarily long.\n\
             It is highly recommended that you install Ninja.",
        );
    }

    config
        .env("LLVM_PREFIX", &llvm_prefix)
        .define("LLVM_DIR", llvm_dir)
        .define("MLIR_DIR", mlir_dir)
        .cxxflag(&format!("-I{}", lumen_llvm_include_dir))
        .configure_arg("-Wno-dev");

    if link_lto == "true" {
        config.cxxflag("-flto=thin");
    }

    if env::var_os("LLVM_NDEBUG").is_some() {
        config.define("NDEBUG", "1");
    }

    if let Ok(sanitizer) = env::var(ENV_LLVM_USE_SANITIZER) {
        config.define("LLVM_USE_SANITIZER", sanitizer);
    }

    let output_path = config.build();
    let search_path = output_path.join("lib");

    // NOTE: Not all of these are necessarily being used at the moment, so eventually
    // we should clean up, but these cover the libs we're likely to want at some point
    link_libs(&[
        "MLIRAMX",
        "MLIRAMXToLLVMIRTranslation",
        "MLIRAMXTransforms",
        "MLIRAffine",
        "MLIRAffineAnalysis",
        "MLIRAffineToStandard",
        "MLIRAffineTransforms",
        "MLIRAffineUtils",
        "MLIRAnalysis",
        "MLIRArithmetic",
        "MLIRArithmeticToLLVM",
        "MLIRArithmeticToSPIRV",
        "MLIRArithmeticTransforms",
        "MLIRArithmeticUtils",
        "MLIRArmNeon",
        "MLIRArmNeon2dToIntr",
        "MLIRArmNeonToLLVMIRTranslation",
        "MLIRArmSVE",
        "MLIRArmSVEToLLVMIRTranslation",
        "MLIRArmSVETransforms",
        "MLIRAsync",
        "MLIRAsyncToLLVM",
        "MLIRAsyncTransforms",
        "MLIRBufferization",
        "MLIRBufferizationToMemRef",
        "MLIRBufferizationTransforms",
        "MLIRCAPIAsync",
        "MLIRCAPIControlFlow",
        "MLIRCAPIConversion",
        "MLIRCAPIDebug",
        //"MLIRCAPIExecutionEngine",
        "MLIRCAPIFunc",
        "MLIRCAPIGPU",
        "MLIRCAPIIR",
        "MLIRCAPIInterfaces",
        "MLIRCAPILLVM",
        "MLIRCAPILinalg",
        "MLIRCAPIPDL",
        "MLIRCAPIQuant",
        "MLIRCAPIRegistration",
        "MLIRCAPISCF",
        "MLIRCAPIShape",
        "MLIRCAPISparseTensor",
        "MLIRCAPITensor",
        "MLIRCAPITransforms",
        "MLIRCallInterfaces",
        "MLIRCastInterfaces",
        "MLIRComplex",
        "MLIRComplexToLLVM",
        "MLIRComplexToStandard",
        "MLIRControlFlow",
        "MLIRControlFlowInterfaces",
        "MLIRControlFlowToLLVM",
        "MLIRControlFlowToSPIRV",
        "MLIRCopyOpInterface",
        "MLIRDLTI",
        "MLIRDataLayoutInterfaces",
        "MLIRDerivedAttributeOpInterface",
        "MLIRDialect",
        "MLIRDialectUtils",
        //"MLIREmitC",
        //"MLIRExecutionEngine",
        "MLIRFunc",
        "MLIRFuncToLLVM",
        "MLIRFuncToSPIRV",
        "MLIRFuncTransforms",
        "MLIRGPUOps",
        "MLIRGPUToGPURuntimeTransforms",
        "MLIRGPUToNVVMTransforms",
        "MLIRGPUToROCDLTransforms",
        "MLIRGPUToSPIRV",
        "MLIRGPUToVulkanTransforms",
        "MLIRGPUTransforms",
        "MLIRIR",
        "MLIRInferTypeOpInterface",
        //"MLIRJitRunner",
        "MLIRLLVMCommonConversion",
        "MLIRLLVMIR",
        "MLIRLLVMIRTransforms",
        "MLIRLLVMToLLVMIRTranslation",
        "MLIRLinalg",
        "MLIRLinalgAnalysis",
        "MLIRLinalgToLLVM",
        "MLIRLinalgToSPIRV",
        "MLIRLinalgToStandard",
        "MLIRLinalgTransforms",
        "MLIRLinalgUtils",
        "MLIRLoopLikeInterface",
        //"MLIRLspServerLib",
        //"MLIRLspServerSupportLib",
        "MLIRMath",
        "MLIRMathToLLVM",
        "MLIRMathToLibm",
        "MLIRMathToSPIRV",
        "MLIRMathTransforms",
        "MLIRMemRef",
        "MLIRMemRefToLLVM",
        "MLIRMemRefToSPIRV",
        "MLIRMemRefTransforms",
        "MLIRMemRefUtils",
        //"MLIRMlirOptMain",
        "MLIRModuleBufferization",
        "MLIRNVVMIR",
        "MLIRNVVMToLLVMIRTranslation",
        "MLIROpenACC",
        "MLIROpenACCToLLVM",
        "MLIROpenACCToLLVMIRTranslation",
        "MLIROpenACCToSCF",
        "MLIROpenMP",
        "MLIROpenMPToLLVM",
        "MLIROpenMPToLLVMIRTranslation",
        //"MLIROptLib",
        "MLIRPDL",
        "MLIRPDLInterp",
        "MLIRPDLLAST",
        "MLIRPDLLCodeGen",
        "MLIRPDLLODS",
        "MLIRPDLToPDLInterp",
        "MLIRParser",
        "MLIRPass",
        "MLIRPresburger",
        "MLIRQuant",
        "MLIRQuantTransforms",
        "MLIRQuantUtils",
        "MLIRROCDLIR",
        "MLIRROCDLToLLVMIRTranslation",
        "MLIRReconcileUnrealizedCasts",
        //"MLIRReduce",
        //"MLIRReduceLib",
        "MLIRRewrite",
        "MLIRSCF",
        "MLIRSCFToControlFlow",
        "MLIRSCFToGPU",
        "MLIRSCFToOpenMP",
        "MLIRSCFToSPIRV",
        "MLIRSCFTransforms",
        "MLIRSCFUtils",
        "MLIRSPIRV",
        "MLIRSPIRVBinaryUtils",
        "MLIRSPIRVConversion",
        "MLIRSPIRVDeserialization",
        "MLIRSPIRVModuleCombiner",
        "MLIRSPIRVSerialization",
        "MLIRSPIRVToLLVM",
        "MLIRSPIRVTransforms",
        "MLIRSPIRVTranslateRegistration",
        "MLIRSPIRVUtils",
        "MLIRShape",
        "MLIRShapeOpsTransforms",
        "MLIRShapeToStandard",
        "MLIRSideEffectInterfaces",
        "MLIRSparseTensor",
        "MLIRSparseTensorPipelines",
        "MLIRSparseTensorTransforms",
        "MLIRSparseTensorUtils",
        "MLIRSupport",
        "MLIRSupportIndentedOstream",
        "MLIRTableGen",
        "MLIRTargetCpp",
        "MLIRTargetLLVMIRExport",
        "MLIRTargetLLVMIRImport",
        "MLIRTensor",
        "MLIRTensorInferTypeOpInterfaceImpl",
        "MLIRTensorTilingInterfaceImpl",
        "MLIRTensorToSPIRV",
        "MLIRTensorTransforms",
        "MLIRTensorUtils",
        "MLIRTilingInterface",
        "MLIRToLLVMIRTranslationRegistration",
        "MLIRTosa",
        "MLIRTosaToArith",
        "MLIRTosaToLinalg",
        "MLIRTosaToSCF",
        "MLIRTosaToTensor",
        "MLIRTosaTransforms",
        "MLIRTransformUtils",
        "MLIRTransforms",
        "MLIRTranslateLib",
        "MLIRVector",
        "MLIRVectorInterfaces",
        "MLIRVectorToGPU",
        "MLIRVectorToLLVM",
        "MLIRVectorToROCDL",
        "MLIRVectorToSCF",
        "MLIRVectorToSPIRV",
        "MLIRVectorTransforms",
        "MLIRVectorUtils",
        "MLIRViewLikeInterface",
        "MLIRX86Vector",
        "MLIRX86VectorToLLVMIRTranslation",
        "MLIRX86VectorTransforms",
    ]);

    println!("cargo:rustc-link-search=native={}", search_path.display());
    println!("cargo:rustc-link-lib=static=MLIRLumenExtensions");
    println!("cargo:rustc-link-lib=static=CIR");
    println!("cargo:rustc-link-lib=static=CIRCAPI");
}

fn link_libs(libs: &[&str]) {
    match env::var_os(ENV_LLVM_LINK_STATIC) {
        Some(val) if val == "true" => link_libs_static(libs),
        _ => link_libs_dylib(libs),
    }
}

#[inline]
fn link_libs_static(libs: &[&str]) {
    for lib in libs {
        link_lib_static(lib);
    }
}

#[inline]
fn link_libs_dylib(libs: &[&str]) {
    let llvm_link_llvm_dylib = env::var(ENV_LLVM_LINK_LLVM_DYLIB).unwrap_or("false".to_owned());
    if llvm_link_llvm_dylib == "true" {
        link_lib_dylib("MLIR");
        link_lib_dylib("MLIR-C");
    } else {
        for lib in libs {
            link_lib_dylib(lib);
        }
    }
}

#[inline]
fn link_lib_static(lib: &str) {
    //println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
    println!("cargo:rustc-link-lib=static={}", lib);
}

#[inline]
fn link_lib_dylib(lib: &str) {
    println!("cargo:rustc-link-lib=dylib={}", lib);
}

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}
