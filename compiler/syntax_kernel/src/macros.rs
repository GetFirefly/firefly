#[macro_export]
macro_rules! kreturn {
    ($span:expr) => {
        Return::new($span, vec![])
    };

    ($span:expr, $($args:expr),*) => {
        Return::new($span, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kbreak {
    ($span:expr) => {
        Break::new($span, vec![])
    };

    ($span:expr, $($args:expr),*) => {
        Break::new($span, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kgoto {
    ($span:expr, $label:expr, $($args:expr),*) => {
        Goto::new($span, $label, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kseq {
    ($span:expr, $arg:expr, $body:expr) => {
        Seq::new($span, $arg, $body)
    };
}

#[macro_export]
macro_rules! kbif {
    ($span:expr, $op:expr, $($args:expr),*) => {
        Bif::new($span, $op, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! ktest {
    ($span:expr, $op:expr, $($args:expr),*) => {
        Test::new($span, $op, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kcall {
    ($span:expr, $callee:expr, $($args:expr),*) => {
        Call::new($span, $callee, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kenter {
    ($span:expr, $callee:expr, $($args:expr),*) => {
        Enter::new($span, $callee, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kput {
    ($span:expr, $arg:expr) => {
        Put::new($span, $arg)
    };
}

#[macro_export]
macro_rules! kcons {
    ($span:expr, $head:expr, $tail:expr) => {
        Cons::new($span, $head, $tail)
    };
}

#[macro_export]
macro_rules! ktuple {
    ($span:expr, $($args:expr),*) => {
        Tuple::new($span, vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kvalues {
    ($($args:expr),*) => {
        IValues::new(vec![$($args,)*])
    }
}

#[macro_export]
macro_rules! kset {
    ($span:expr, $var:expr, $arg:expr) => {
        ISet {
            span: $span,
            annotations: Annotations::default(),
            vars: vec![$var],
            arg: Box::new($arg),
            body: None,
        }
    };
}
