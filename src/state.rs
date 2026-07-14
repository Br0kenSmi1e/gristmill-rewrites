//! Symbolic rewrite state.

/// A canonical symbolic computation together with its protected outputs.
///
/// The representation is intentionally opaque until the tensor computation
/// model and state invariants are designed.
pub struct State {
    _private: (),
}
