pub mod parse;

use std::borrow::Cow;

use failure::Error;

pub trait AstPass {
    type Output;

    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run(&self, compiler: &mut Compiler, ast: &mut Module) -> Result<(), Error>;
}

pub fn run_ast_passes(compiler: &mut Compiler, ast: &mut Module, passes: &[&dyn AstPass]) -> Result<(), Error> {
    for pass in passes {
        let name = pass.name();
        compiler.write_debug(format!("Running pass {}", name));
        pass.run(compiler, ast)?;
        compiler.write_debug(format!("{} finished in {}ms", name));
    }

    Ok(())
}

/// Generates a default name for the pass based on the name of the
/// type `T`.
fn default_name<T: ?Sized>() -> Cow<'static, str> {
    let name = unsafe { ::std::intrinsics::type_name::<T>() };
    if let Some(tail) = name.rfind(":") {
        Cow::from(&name[tail+1..])
    } else {
        Cow::from(name)
    }
}
