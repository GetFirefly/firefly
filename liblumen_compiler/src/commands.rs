pub(crate) mod compile;
pub(crate) mod print;

use std::sync::{Arc, Mutex};

use libeir_diagnostics::{CodeMap, Emitter};

use liblumen_session::{DiagnosticsConfig, DiagnosticsHandler, Options};

pub(super) fn default_diagnostics_handler(
    options: &Options,
    emitter: Option<Arc<dyn Emitter>>,
) -> DiagnosticsHandler {
    let codemap = Arc::new(Mutex::new(CodeMap::new()));
    create_diagnostics_handler(options, codemap, emitter)
}

pub(super) fn create_diagnostics_handler(
    options: &Options,
    codemap: Arc<Mutex<CodeMap>>,
    emitter: Option<Arc<dyn Emitter>>,
) -> DiagnosticsHandler {
    let emitter = emitter.unwrap_or_else(|| default_emitter(codemap.clone(), &options));
    let config = DiagnosticsConfig {
        warnings_as_errors: options.warnings_as_errors,
        no_warn: options.no_warn,
    };
    DiagnosticsHandler::new(config, codemap, emitter)
}

pub(super) fn default_emitter(codemap: Arc<Mutex<CodeMap>>, options: &Options) -> Arc<dyn Emitter> {
    use libeir_diagnostics::{NullEmitter, StandardStreamEmitter};
    use liblumen_session::verbosity_to_severity;
    use liblumen_util::error::Verbosity;

    match options.verbosity {
        Verbosity::Silent => Arc::new(NullEmitter::new()),
        v => Arc::new(
            StandardStreamEmitter::new(options.use_color.into())
                .set_codemap(codemap)
                .set_min_severity(verbosity_to_severity(v)),
        ),
    }
}

pub(super) fn abort_on_err<T>(_: ()) -> T {
    use liblumen_util::error::FatalError;

    FatalError.raise()
}
