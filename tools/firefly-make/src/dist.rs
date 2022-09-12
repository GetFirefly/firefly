use std::fs::File;
use std::path::PathBuf;

use cargo_metadata::MetadataCommand;
use clap::Args;
use flate2::write::GzEncoder;
use flate2::Compression;

#[derive(Args)]
pub struct Config {
    /// The working directory for the build
    #[clap(hide(true), long, env("CARGO_MAKE_WORKING_DIRECTORY"))]
    cwd: Option<PathBuf>,
    /// The name of the target platform to build for
    #[clap(long, env("CARGO_MAKE_RUST_TARGET_TRIPLE"))]
    target_triple: String,
    /// The location where the compiler toolchain should be installed
    #[clap(long, env("FIREFLY_INSTALL_DIR"), default_value = "./_build")]
    install_dir: PathBuf,
    #[clap(long, env("FIREFLY_DIST_DIR"), default_value = "./")]
    output_dir: PathBuf,
    #[clap(long, env("FIREFLY_VERSION"))]
    version: Option<String>,
}
impl Config {
    pub fn working_directory(&self) -> PathBuf {
        self.cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap())
    }
}

pub fn run(config: &Config) -> anyhow::Result<()> {
    let cwd = config.working_directory();
    let install_dir = config.install_dir.as_path();
    let output_dir = config.output_dir.as_path();

    println!("Detecting release metadata..");

    let version = if let Some(version) = config.version.as_ref() {
        version.clone()
    } else {
        let metadata = MetadataCommand::new()
            .manifest_path(cwd.join("firefly").join("Cargo.toml"))
            .no_deps()
            .exec()
            .unwrap();
        metadata
            .packages
            .iter()
            .find_map(|p| {
                if p.name == "firefly" {
                    Some(p.version.to_string())
                } else {
                    None
                }
            })
            .expect("could not find version information for `firefly`")
    };

    let filename = format!("firefly-{}-{}.tar.gz", &version, &config.target_triple);
    let output_file = output_dir.join(&filename);

    println!("Packaging release to {}", output_file.display());

    // Open the .tar.gz file for writing
    let mut output_file = File::options()
        .write(true)
        .create(true)
        .open(output_dir.join(&filename))?;
    {
        // Wrap the file stream in a gzip encoder
        let mut encoder = GzEncoder::new(&mut output_file, Compression::default());
        {
            // Construct the archive containing the distribution contents, laid out such that extracting the tarball to a directory
            // unpacks the following layout:
            //
            //     bin/
            //       firefly[.exe]
            //     etc/
            //     lib/
            //       fireflylib/
            //         <llvm_target>/
            //     libexec/
            //     share/
            //
            let mut builder = tar::Builder::new(&mut encoder);
            builder.follow_symlinks(false);
            println!(
                "Adding artifacts from install directory at {}",
                install_dir.display()
            );
            builder.append_dir_all("bin", install_dir.join("bin"))?;
            builder.append_dir_all("lib", install_dir.join("lib"))?;
            builder.finish()?;
        }
        encoder.finish()?;
    }

    println!("Packaging complete!");

    Ok(())
}
