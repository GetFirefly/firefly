extern crate cmake;
extern crate handlebars;
extern crate heck;
extern crate serde;
extern crate serde_json;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ENV_LLVM_CORE_INCLUDE: &'static str = "DEP_LUMEN_LLVM_CORE_INCLUDE";
const ENV_LLVM_PREFIX: &'static str = "DEP_LUMEN_LLVM_CORE_PREFIX";
const ENV_LLVM_LINK_STATIC: &'static str = "DEP_LUMEN_LLVM_CORE_LINK_STATIC";
const ENV_LLVM_LINK_LLVM_DYLIB: &'static str = "DEP_LUMEN_LLVM_CORE_LINK_LLVM_DYLIB";
const ENV_LLVM_LTO: &'static str = "DEP_LUMEN_LLVM_CORE_LTO";

mod types {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tablegen/types.rs"
    ));
}

mod codegen {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tablegen/codegen.rs"
    ));
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap().to_path_buf();
    let liblumen_term_dir = workspace_dir.join("term");
    let lumen_term_include_dir = liblumen_term_dir.join("c_src/include");
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

    // Generate dialect-specific types/builders/trait impls for dialects we use
    //generate_dialects(llvm_prefix.as_path(), lumen_term_include_dir.as_path());

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
        .cxxflag(&format!("-I{}", lumen_term_include_dir.display()))
        .configure_arg("-Wno-dev");

    if link_lto == "true" {
        config.cxxflag("-flto=thin");
    }

    if env::var_os("LLVM_NDEBUG").is_some() {
        config.define("NDEBUG", "1");
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

struct DialectInfo {
    name: &'static str,
    path: PathBuf,
}

#[allow(dead_code)]
fn generate_dialects(llvm_prefix: &Path, liblumen_term_include_dir: &Path) {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mlir_include =
        PathBuf::from(env::current_dir().expect("unable to access current directory"))
            .join("c_src/include");
    let llvm_include = llvm_prefix.join("include");
    let llvm_tblgen = llvm_prefix.join("bin/llvm-tblgen");

    // The following are the dialects containing types/ops we use
    let dialects = vec![
        DialectInfo {
            name: "cir",
            path: mlir_include.join("CIR/CIR.td"),
        },
        DialectInfo {
            name: "func",
            path: llvm_prefix.join("include/mlir/Dialect/Func/IR/FuncOps.td"),
        },
        DialectInfo {
            name: "cf",
            path: llvm_prefix.join("include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td"),
        },
        DialectInfo {
            name: "arith",
            path: llvm_prefix.join("include/mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"),
        },
        DialectInfo {
            name: "llvmir",
            path: llvm_prefix.join("include/mlir/Dialect/LLVMIR/LLVMOps.td"),
        },
    ];

    for dialect in dialects.iter() {
        let filename = format!("{}.tblgen.json", dialect.name);
        let outfile = out_dir.join(filename.as_str());
        let mut tblgen = Command::new(llvm_tblgen.as_path());
        let tblgen = tblgen
            .current_dir(out_dir.as_path())
            .arg("--dump-json")
            .arg("-o")
            .arg(outfile.as_path())
            .arg("-I")
            .arg(llvm_include.as_path())
            .arg("-I")
            .arg(liblumen_term_include_dir)
            .arg("-I")
            .arg(mlir_include.as_path())
            .arg(dialect.path.as_path());

        let _ = output(tblgen);

        generate_dialect(out_dir.as_path(), outfile, dialect);
    }

    {
        let generated_file = PathBuf::from(out_dir.join("dialects.rs"));
        let mut generated = File::create(&generated_file).unwrap();
        for DialectInfo { ref name, .. } in dialects.iter() {
            writeln!(
                generated,
                r#"
pub mod {} {{
    include!(concat!(env!("OUT_DIR"), "/{}.rs"));
}}
"#,
                name, name
            )
            .unwrap();
        }
    }
}

const JQ_SCRIPT: &'static str = r#"
. as $root | (."!instanceof".Dialect | map($root[.]) | .[0]) + {
  attributes: (."!instanceof".AttrDef | map($root[.])),
  types: (."!instanceof".TypeDef | map($root[.])),
  operations: (."!instanceof".Op | map($root[.]) | map(.successors.args |= (. | map({name: .[1]} + $root[.[0].def]))) | map(.traits |= (. | map(. + $root[.def])) )),
  enums: (."!instanceof".IntEnumAttr | map($root[.]) | map(.enumerants |= (. | map(. + $root[.def])))),
  bitflags: (."!instanceof".BitEnumAttr | map($root[.]) | map(.enumerants |= (. | map(. + $root[.def])))),
}
"#;

fn preprocess_dialect(input: &Path) -> String {
    let mut jq = Command::new("jq");
    let jq = jq
        .arg("-r")
        .arg(JQ_SCRIPT)
        .arg(input)
        .stdout(Stdio::piped());
    output(jq)
}

fn generate_dialect(out_dir: &Path, input: PathBuf, dialect: &DialectInfo) {
    // Preprocess the raw TableGen json into a form that contains only what we need
    let json = preprocess_dialect(input.as_path());
    // Construct the output path for the generated Rust module
    let filename = format!("{}.rs", dialect.name);
    let outfile = PathBuf::from(out_dir.join(&filename));
    // Generate the Rust module
    let dialect: types::Dialect = serde_json::from_str(&json).unwrap();
    let mut file = File::create(&outfile).unwrap();
    codegen::to_rust(dialect, &mut file).unwrap();
}

fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!(
            "failed to execute command: {:?}\nerror: {}",
            cmd, e
        )),
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap()
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

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}
