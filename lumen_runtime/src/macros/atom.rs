macro_rules! boolean_infix_operator {
    ($left:ident, $right:ident, $operator:tt) => {{
        use std::convert::TryInto;

        use liblumen_alloc::erts::term::prelude::Encoded;

        let left_bool: bool = $left.decode()?.try_into()?;
        let right_bool: bool = $right.decode()?.try_into()?;
        let output_bool = left_bool $operator right_bool;

        Ok(output_bool.into())
    }};
}
