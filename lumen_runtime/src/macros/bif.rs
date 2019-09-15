macro_rules! bif {
  ($module:expr, $function:literal, $native:ident, $($arg:ident),*) => {
    module_function_arity!($module, $function, count!($($arg)*));
    code!($native, $($arg),*);
    frame!(module_function_arity(), code);
    place_frame_with_arguments!(frame(), $($arg),*);
  };
}

macro_rules! count {
    () => {0usize};
    ($arg:ident $($args:ident)*) => {
        1usize + count!($($args)*)
    }
}

macro_rules! place_frame_with_arguments {
    ($frame:expr, $($arg:ident),*) => {
        pub fn place_frame_with_arguments(
            process: &Process,
            placement: Placement,
            $($arg: Term),*
        ) -> Result<(), Alloc> {
            push_args!(process, $($arg),*);
            process.place_frame($frame, placement);

            Ok(())
        }
    };
}

macro_rules! push_args {
    () => {()};
    ($process:ident, $arg:ident) => {
      $process.stack_push($arg)?;
    };
    ($process:ident, $arg:ident, $($args:ident),+) => {
        push_args!($process, $($args),+);
        push_args!($process, $arg);
    }
}

macro_rules! code {
    ($native:ident, $($arg:ident),*) => {
        pub(in crate::otp) fn code(arc_process: &Arc<Process>) -> code::Result {
            arc_process.reduce();

            $(
              let $arg = arc_process.stack_pop().unwrap();
            )*

            match $native(arc_process, $($arg),*) {
                Ok(value) => {
                    arc_process.return_from_call(value)?;

                    Process::call_code(arc_process)
                }
                Err(exception) => result_from_exception(arc_process, exception),
            }
        }
    };
}

macro_rules! frame {
    ($mfa:expr, $code:ident) => {
        fn frame() -> Frame {
            Frame::new($mfa, $code)
        }
    };
}

macro_rules! module_function_arity {
    ($module:expr, $function:literal, $arity:expr) => {
        fn function() -> Atom {
            Atom::try_from_str($function).unwrap()
        }

        fn module_function_arity() -> Arc<ModuleFunctionArity> {
            Arc::new(ModuleFunctionArity {
                module: $module,
                function: function(),
                arity: $arity,
            })
        }
    };
}
