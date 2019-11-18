use std::fmt;
use std::convert::AsRef;
use std::path::{Path, PathBuf};

use super::llvm;
use super::llvm::enums::*;
use super::llvm::target::{Target, TargetData, TargetMachine};

use super::{Result, CodeGenError};

/// The configuration builder used to configure a target for code generation.
/// 
/// This is intended for use by consumers of the codegen API. Once a config is
/// built (producing a `Config`), it is static and cannot be changed.
#[derive(Clone)]
pub struct ConfigBuilder {
    target_triple: String,
    target_features: String,
    target_cpu: String,
    target: Target,
    opt_level: OptimizationLevel,
    relocation_mode: RelocMode,
    code_model: CodeModel,
    output_type: OutputType,
    build_dir: PathBuf,
    output_dir: Option<PathBuf>,
}
impl ConfigBuilder {
    pub fn new<T: AsRef<str>>(triple: T) -> Result<Self> {
        let s = triple.as_ref();
        match llvm::target::from_triple(s) {
            Err(err) => Err(CodeGenError::InvalidTarget(err.to_string())),
            Ok(target) => {
                let build_dir = Self::default_build_dir(s);
                Ok(Self {
                    target_triple: s.to_string(),
                    target_features: "".to_owned(),
                    target_cpu: "".to_owned(),
                    target,
                    opt_level: OptimizationLevel::Default,
                    relocation_mode: RelocMode::Default,
                    code_model: CodeModel::Default,
                    output_type: OutputType::Assembly,
                    build_dir,
                    output_dir: None,
                })
            }
        }
    }

    fn default_build_dir(triple: &str) -> PathBuf {
        let cwd = std::env::current_dir()
            .expect("The current directory is inaccessible!");
        let target_dir = &format!("_build/lumen/target/{}", triple);
        cwd.join(Path::new(target_dir))
    }

    #[inline]
    pub fn set_output_dir(mut self, output_dir: PathBuf) -> Self {
        self.output_dir = output_dir;
        self
    }

    #[inline]
    pub fn set_target_features(mut self, features: &str) -> Self {
        self.target_features = features.to_owned();
        self
    }

    #[inline]
    pub fn set_target_cpu(mut self, cpu: &str) -> Self {
        self.target_cpu = cpu.to_owned();
        self
    }

    #[inline]
    pub fn set_optimization_level(mut self, opt: OptimizationLevel) -> Self {
        self.opt_level = opt;
        self
    }

    #[inline]
    pub fn set_relocation_mode(mut self, mode: RelocMode) -> Self {
        self.relocation_mode = mode;
        self
    }

    #[inline]
    pub fn set_code_model(mut self, model: CodeModel) -> Self {
        self.code_model = model;
        self
    }

    #[inline]
    pub fn set_output_type(mut self, ty: OutputType) -> Self {
        self.output_type = ty;
        self
    }

    #[inline]
    pub fn finalize(self) -> Result<Config> {
        if self.target.has_target_machine() {
            Config::new(self)
        } else {
            Err(CodeGenError::no_target_machine(self.target_triple, self.target_cpu, self.target_features))
        }
    }
}
impl fmt::Debug for ConfigBuilder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ConfigBuilder")
            .field("target_triple", &self.target_triple)
            .field("target_features", &self.target_features)
            .field("target_cpu", &self.target_cpu)
            .field("opt_level", &self.opt_level)
            .field("relocation_mode", &self.relocation_mode)
            .field("code_model", &self.code_model)
            .field("output_type", &self.output_type)
            .field("build_dir", &self.build_dir)
            .field("output_dir", &self.output_dir)
            .finish()
    }
}
impl Default for ConfigBuilder {
    fn default() -> Self {
        let build_dir = Self::default_build_dir();
        Self {
            target_triple: llvm::target::default_triple(),
            target_features: llvm::target::host_features(),
            target_cpu: llvm::target::host_cpu(),
            target: llvm::target::current(),
            opt_level: OptimizationLevel::Default,
            relocation_mode: RelocMode::Default,
            code_model: CodeModel::Default,
            output_type: OutputType::Assembly,
            build_dir,
            output_dir: None,
        }
    }
}

/// This struct represents the finalized target configuration
pub struct Config {
    target: Target,
    target_machine: TargetMachine,
    target_data: TargetData,
    triple: String,
    opt_level: OptimizationLevel,
    relocation_mode: RelocMode,
    code_model: CodeModel,
    output_type: OutputType,
    build_dir: PathBuf,
    output_dir: PathBuf,
}
impl Config {
    pub(in super) fn new(target_config: ConfigBuilder) -> Result<Self> {
        let triple = target_config.target_triple;
        let cpu = target_config.target_cpu;
        let features = target_config.target_features;
        let opt_level = target_config.opt_level;
        let relocation_mode = target_config.relocation_mode;
        let code_model = target_config.code_model;
        let output_type = target_config.output_type;
        let target = target_config.target;
        let machine = target.create_target_machine(
            triple.as_str(), 
            cpu.as_str(), 
            features.as_str(), 
            opt_level, 
            relocation_mode, 
            code_model
        );
        match machine {
            None => Err(CodeGenError::no_target_machine(triple, cpu, features)),
            Some(target_machine) => {
                let target_data = target_machine.get_target_data();
                Self {
                    target,
                    target_machine,
                    target_data,
                    triple,
                    opt_level,
                    relocation_mode,
                    code_model,
                    output_type,
                    build_dir: target_config.build_dir,
                    output_dir: target_config.output_dir,
                }
            }
        }
    }

    #[inline]
    pub fn target(&self) -> &Target {
        &self.target
    }

    #[inline]
    pub fn target_machine(&self) -> &TargetMachine {
        &self.target_machine
    }

    #[inline]
    pub fn target_data(&self) -> &TargetData {
        &self.target_data
    }

    #[inline]
    pub fn target_triple(&self) -> &str {
        self.triple.as_str()
    }

    #[inline]
    pub fn target_arch(&self) -> &str {
        self.triple.split('-').next().unwrap()
    }

    #[inline]
    pub fn opt_level(&self) -> OptimizationLevel {
        self.opt_level
    }

    #[inline]
    pub fn relocation_mode(&self) -> RelocMode {
        self.relocation_mode
    }

    #[inline]
    pub fn code_model(&self) -> CodeModel {
        self.code_model
    }

    #[inline]
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    #[inline]
    pub fn build_dir(&self) -> &Path {
        self.build_dir.as_path()
    }

    #[inline]
    pub fn output_dir(&self) -> &Path {
        self.output_dir.as_path()
    }
}
