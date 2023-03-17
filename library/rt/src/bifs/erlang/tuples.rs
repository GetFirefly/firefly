use firefly_alloc::heap::Heap;

use smallvec::SmallVec;

use crate::function::ErlangResult;
use crate::gc::{garbage_collect, RootSet};
use crate::process::ProcessLock;
use crate::term::*;

#[export_name = "erlang:is_tuple/1"]
pub extern "C" fn is_tuple(_process: &mut ProcessLock, tuple: OpaqueTerm) -> ErlangResult {
    ErlangResult::Ok(tuple.is_tuple().into())
}

#[export_name = "erlang:tuple_size/1"]
pub extern "C-unwind" fn tuple_size1(process: &mut ProcessLock, tuple: OpaqueTerm) -> ErlangResult {
    match tuple.tuple_size() {
        Ok(arity) => ErlangResult::Ok(Term::Int(arity as i64).into()),
        Err(_) => badarg!(process, tuple),
    }
}

#[export_name = "erlang:element/2"]
pub extern "C" fn element2(
    process: &mut ProcessLock,
    index: OpaqueTerm,
    tuple_term: OpaqueTerm,
) -> ErlangResult {
    let Ok(index) = OneBasedIndex::try_from(index) else { badarg!(process, index); };
    let Term::Tuple(tuple) = tuple_term.into() else { badarg!(process, tuple_term); };
    match tuple.get_element(index) {
        Some(element) => ErlangResult::Ok(element),
        None => badarg!(process, tuple_term),
    }
}

#[export_name = "erlang:setelement/3"]
pub extern "C" fn setelement3(
    process: &mut ProcessLock,
    index_term: OpaqueTerm,
    tuple_term: OpaqueTerm,
    value: OpaqueTerm,
) -> ErlangResult {
    let Ok(index) = OneBasedIndex::try_from(index_term) else { badarg!(process, index_term); };
    let Term::Tuple(tuple) = tuple_term.into() else { badarg!(process, tuple_term); };
    if index > tuple.len() {
        badarg!(process, index_term);
    }

    match tuple.set_element(index, value, process) {
        Ok(new_tuple) => ErlangResult::Ok(new_tuple.into()),
        Err(_) => {
            let mut roots = RootSet::default();
            let mut tuple = Term::Tuple(tuple);
            roots += &mut tuple as *mut Term;
            assert!(garbage_collect(process, roots).is_ok());
            let Term::Tuple(tuple) = tuple else { unreachable!() };
            ErlangResult::Ok(tuple.set_element(index, value, process).unwrap().into())
        }
    }
}

#[export_name = "erlang:tuple_to_list/1"]
pub extern "C-unwind" fn tuple_to_list1(
    process: &mut ProcessLock,
    mut tuple: OpaqueTerm,
) -> ErlangResult {
    let Ok(arity) = tuple.tuple_size() else { badarg!(process, tuple); };

    let mut layout = LayoutBuilder::new();
    layout.build_list(arity as usize);
    let needed = layout.finish().size();
    if needed > process.heap_available() {
        let mut roots = RootSet::default();
        roots += &mut tuple as *mut OpaqueTerm;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let Term::Tuple(tuple) = tuple.into() else { unreachable!() };
    let mut builder = ListBuilder::new(process);
    for term in tuple.as_slice() {
        unsafe {
            builder.push_unsafe(*term).unwrap();
        }
    }
    ErlangResult::Ok(builder.finish().map(Term::Cons).unwrap_or(Term::Nil).into())
}

#[export_name = "erlang:make_tuple/2"]
pub extern "C-unwind" fn make_tuple2(
    process: &mut ProcessLock,
    arity: OpaqueTerm,
    initial_value: OpaqueTerm,
) -> ErlangResult {
    make_tuple3(process, arity, initial_value, OpaqueTerm::NIL)
}

#[export_name = "erlang:make_tuple/3"]
pub extern "C-unwind" fn make_tuple3(
    process: &mut ProcessLock,
    arity: OpaqueTerm,
    mut default_value: OpaqueTerm,
    mut init_list: OpaqueTerm,
) -> ErlangResult {
    let arity = usize_or_badarg!(process, arity.into());
    if !init_list.is_list() {
        badarg!(process, init_list);
    }

    let mut layout = LayoutBuilder::new();
    layout.build_tuple(arity);
    let needed = layout.finish().size();
    if needed > process.heap_available() {
        let mut roots = RootSet::default();
        roots += &mut default_value as *mut OpaqueTerm;
        roots += &mut init_list as *mut OpaqueTerm;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let heap_top = process.heap_top();
    let mut tuple = Tuple::new_in(arity, process).unwrap();
    tuple.as_mut_slice().fill(default_value);
    if let Term::Cons(init) = init_list.into() {
        for init_term in init.iter() {
            match init_term {
                Ok(Term::Tuple(init_tuple)) if init_tuple.len() == 2 => {
                    match OneBasedIndex::try_from(init_tuple[0]) {
                        Ok(index) => {
                            tuple[index] = init_tuple[1];
                        }
                        Err(_) => {
                            unsafe {
                                process.reset_heap_top(heap_top);
                            }
                            badarg!(process, init_list);
                        }
                    }
                }
                _ => {
                    unsafe {
                        process.reset_heap_top(heap_top);
                    }
                    badarg!(process, init_list);
                }
            }
        }
    }

    ErlangResult::Ok(tuple.into())
}

#[export_name = "erlang:list_to_tuple/1"]
pub extern "C-unwind" fn list_to_tuple1(
    process: &mut ProcessLock,
    list: OpaqueTerm,
) -> ErlangResult {
    match list.into() {
        Term::Nil => {
            let mut layout = LayoutBuilder::new();
            layout.build_tuple(0);
            let needed = layout.finish().size();
            if needed > process.heap_available() {
                assert!(garbage_collect(process, Default::default()).is_ok());
            }
            let tuple = Tuple::new_in(0, process).unwrap();
            ErlangResult::Ok(tuple.into())
        }
        Term::Cons(cons) => {
            let mut items = SmallVec::<[OpaqueTerm; 8]>::default();
            for maybe_improper in cons.iter_raw() {
                match maybe_improper {
                    Ok(term) => {
                        items.push(term);
                    }
                    Err(_) => badarg!(process, list),
                }
            }
            let mut layout = LayoutBuilder::new();
            layout.build_tuple(items.len());
            let needed = layout.finish().size();
            if needed > process.heap_available() {
                let mut roots = RootSet::default();
                for item in items.iter_mut() {
                    roots += item as *mut OpaqueTerm;
                }
                assert!(garbage_collect(process, roots).is_ok());
            }
            let tuple = Tuple::from_slice(items.as_slice(), process).unwrap();
            ErlangResult::Ok(tuple.into())
        }
        _ => badarg!(process, list),
    }
}
