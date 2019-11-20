macro_rules! boolean_infix_operator {
    ($process: ident, $left:ident, $right:ident, $operator:tt) => {{
        use std::convert::TryInto;

        use anyhow::*;

        use liblumen_alloc::erts::term::prelude::Encoded;

        let left_bool: bool = $left.decode()?.try_into().context("left must be a bool")?;
        let right_bool: bool = $right.decode()?.try_into().context("right must be a bool")?;
        let output_bool = left_bool $operator right_bool;

        Ok(output_bool.into())
    }};
}
