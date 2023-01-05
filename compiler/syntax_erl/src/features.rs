use std::collections::BTreeMap;

use firefly_intern::{symbols, Symbol};

use lazy_static::lazy_static;

lazy_static! {
    static ref FEATURES: Vec<Feature> = {
        vec![Feature::experimental(
            symbols::MaybeExpr,
            "Value based error handling (EEP49)",
            25,
            false,
        )]
    };
}

lazy_static! {
    static ref FEATURE_MAP: BTreeMap<Symbol, &'static Feature> = {
        let mut features = BTreeMap::new();
        for feat in FEATURES.iter() {
            let name = feat.name;
            features.insert(name, feat);
        }
        features
    };
}

/// Describes a selectable feature, corresponding to the BEAM compiler's functionality of the same
/// name
#[derive(Debug, Copy, Clone)]
pub struct Feature {
    pub name: Symbol,
    pub description: &'static str,
    pub initial_otp_release: usize,
    pub enabled: bool,
    pub experimental: bool,
}
impl Feature {
    #[allow(unused)]
    pub fn new(
        name: Symbol,
        description: &'static str,
        initial_otp_release: usize,
        enabled: bool,
    ) -> Self {
        Self {
            name,
            description,
            initial_otp_release,
            enabled,
            experimental: false,
        }
    }

    pub fn experimental(
        name: Symbol,
        description: &'static str,
        initial_otp_release: usize,
        enabled: bool,
    ) -> Self {
        Self {
            name,
            description,
            initial_otp_release,
            enabled,
            experimental: true,
        }
    }
}

/// Get the feature matching the provided name
pub fn get(op: &Symbol) -> Option<&'static Feature> {
    FEATURE_MAP.get(op).copied()
}
