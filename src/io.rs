//! JSON import and export for validated rewrite states.

use crate::{
    repr::{Computation, TensorId},
    state::{State, StateError},
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct StateJson {
    computation: Computation,
    protected_outputs: Vec<TensorId>,
}

#[derive(Serialize)]
struct StateJsonRef<'a> {
    computation: &'a Computation,
    protected_outputs: &'a [TensorId],
}

/// A filesystem, JSON, or state-validation failure during import or export.
#[derive(Debug)]
pub enum IoJsonError {
    Io(std::io::Error),
    Json(serde_json::Error),
    State(StateError),
}

impl From<std::io::Error> for IoJsonError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for IoJsonError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<StateError> for IoJsonError {
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}

impl fmt::Display for IoJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::State(error) => write!(formatter, "invalid state: {error:?}"),
        }
    }
}

impl Error for IoJsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::State(_) => None,
        }
    }
}

/// Read and validate a rewrite state from a UTF-8 JSON file.
pub fn read_json(path: impl AsRef<Path>) -> Result<State, IoJsonError> {
    from_json(&fs::read_to_string(path)?)
}

/// Write a rewrite state as pretty JSON.
pub fn write_json(path: impl AsRef<Path>, state: &State) -> Result<(), IoJsonError> {
    Ok(fs::write(path, to_json(state)?)?)
}

/// Parse, validate, and canonicalize a rewrite state from JSON text.
pub fn from_json(input: &str) -> Result<State, IoJsonError> {
    let state: StateJson = serde_json::from_str(input)?;
    Ok(State::new(state.computation, state.protected_outputs)?)
}

/// Serialize a rewrite state as pretty JSON.
pub fn to_json(state: &State) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&StateJsonRef {
        computation: state.computation(),
        protected_outputs: state.protected_outputs(),
    })
}

pub(crate) mod coefficient {
    use crate::repr::Coefficient;
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(coefficient: &Coefficient, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&coefficient.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Coefficient, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(D::Error::custom)
    }
}
