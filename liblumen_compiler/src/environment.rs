use std::collections::HashSet;

use lazy_static::lazy_static;

use liblumen_syntax::Symbol;

lazy_static! {
    pub static ref BIFS: HashSet<(Symbol, usize)> = {
        let bifs = HashSet::new();
        bifs.insert((Symbol::intern("abs"), 1));
        bifs.insert((Symbol::intern("apply"), 2));
        bifs.insert((Symbol::intern("apply"), 3));
        bifs.insert((Symbol::intern("atom_to_binary"), 2));
        bifs.insert((Symbol::intern("atom_to_list"), 1));
        bifs.insert((Symbol::intern("binary_part"), 2));
        bifs.insert((Symbol::intern("binary_part"), 3));
        bifs.insert((Symbol::intern("binary_to_atom"), 2));
        bifs.insert((Symbol::intern("binary_to_existing_atom"), 2));
        bifs.insert((Symbol::intern("binary_to_integer"), 1));
        bifs.insert((Symbol::intern("binary_to_integer"), 2));
        bifs.insert((Symbol::intern("binary_to_float"), 1));
        bifs.insert((Symbol::intern("binary_to_list"), 1));
        bifs.insert((Symbol::intern("binary_to_list"), 3));
        bifs.insert((Symbol::intern("binary_to_term"), 1));
        bifs.insert((Symbol::intern("binary_to_term"), 2));
        bifs.insert((Symbol::intern("bitsize"), 1));
        bifs.insert((Symbol::intern("bit_size"), 1));
        bifs.insert((Symbol::intern("bitstring_to_list"), 1));
        bifs.insert((Symbol::intern("byte_size"), 1));
        bifs.insert((Symbol::intern("ceil"), 1));
        bifs.insert((Symbol::intern("check_old_code"), 1));
        bifs.insert((Symbol::intern("check_process_code"), 2));
        bifs.insert((Symbol::intern("check_process_code"), 3));
        bifs.insert((Symbol::intern("date"), 0));
        bifs.insert((Symbol::intern("delete_module"), 1));
        bifs.insert((Symbol::intern("demonitor"), 1));
        bifs.insert((Symbol::intern("demonitor"), 2));
        bifs.insert((Symbol::intern("disconnect_node"), 1));
        bifs.insert((Symbol::intern("element"), 2));
        bifs.insert((Symbol::intern("erase"), 0));
        bifs.insert((Symbol::intern("erase"), 1));
        bifs.insert((Symbol::intern("error"), 1));
        bifs.insert((Symbol::intern("error"), 2));
        bifs.insert((Symbol::intern("exit"), 1));
        bifs.insert((Symbol::intern("exit"), 2));
        bifs.insert((Symbol::intern("float"), 1));
        bifs.insert((Symbol::intern("float_to_list"), 1));
        bifs.insert((Symbol::intern("float_to_list"), 2));
        bifs.insert((Symbol::intern("float_to_binary"), 1));
        bifs.insert((Symbol::intern("float_to_binary"), 2));
        bifs.insert((Symbol::intern("floor"), 1));
        bifs.insert((Symbol::intern("garbage_collect"), 0));
        bifs.insert((Symbol::intern("garbage_collect"), 1));
        bifs.insert((Symbol::intern("garbage_collect"), 2));
        bifs.insert((Symbol::intern("get"), 0));
        bifs.insert((Symbol::intern("get"), 1));
        bifs.insert((Symbol::intern("get_keys"), 0));
        bifs.insert((Symbol::intern("get_keys"), 1));
        bifs.insert((Symbol::intern("group_leader"), 0));
        bifs.insert((Symbol::intern("group_leader"), 2));
        bifs.insert((Symbol::intern("halt"), 0));
        bifs.insert((Symbol::intern("halt"), 1));
        bifs.insert((Symbol::intern("halt"), 2));
        bifs.insert((Symbol::intern("hd"), 1));
        bifs.insert((Symbol::intern("integer_to_binary"), 1));
        bifs.insert((Symbol::intern("integer_to_binary"), 2));
        bifs.insert((Symbol::intern("integer_to_list"), 1));
        bifs.insert((Symbol::intern("integer_to_list"), 2));
        bifs.insert((Symbol::intern("iolist_size"), 1));
        bifs.insert((Symbol::intern("iolist_to_binary"), 1));
        bifs.insert((Symbol::intern("is_alive"), 0));
        bifs.insert((Symbol::intern("is_process_alive"), 1));
        bifs.insert((Symbol::intern("is_atom"), 1));
        bifs.insert((Symbol::intern("is_boolean"), 1));
        bifs.insert((Symbol::intern("is_binary"), 1));
        bifs.insert((Symbol::intern("is_bitstr"), 1));
        bifs.insert((Symbol::intern("is_bitstring"), 1));
        bifs.insert((Symbol::intern("is_float"), 1));
        bifs.insert((Symbol::intern("is_function"), 1));
        bifs.insert((Symbol::intern("is_function"), 2));
        bifs.insert((Symbol::intern("is_integer"), 1));
        bifs.insert((Symbol::intern("is_list"), 1));
        bifs.insert((Symbol::intern("is_map"), 1));
        bifs.insert((Symbol::intern("is_map_key"), 2));
        bifs.insert((Symbol::intern("is_number"), 1));
        bifs.insert((Symbol::intern("is_pid"), 1));
        bifs.insert((Symbol::intern("is_port"), 1));
        bifs.insert((Symbol::intern("is_reference"), 1));
        bifs.insert((Symbol::intern("is_tuple"), 1));
        bifs.insert((Symbol::intern("is_record"), 2));
        bifs.insert((Symbol::intern("is_record"), 3));
        bifs.insert((Symbol::intern("length"), 1));
        bifs.insert((Symbol::intern("link"), 1));
        bifs.insert((Symbol::intern("list_to_atom"), 1));
        bifs.insert((Symbol::intern("list_to_binary"), 1));
        bifs.insert((Symbol::intern("list_to_bitstring"), 1));
        bifs.insert((Symbol::intern("list_to_existing_atom"), 1));
        bifs.insert((Symbol::intern("list_to_float"), 1));
        bifs.insert((Symbol::intern("list_to_integer"), 1));
        bifs.insert((Symbol::intern("list_to_integer"), 2));
        bifs.insert((Symbol::intern("list_to_pid"), 1));
        bifs.insert((Symbol::intern("list_to_port"), 1));
        bifs.insert((Symbol::intern("list_to_ref"), 1));
        bifs.insert((Symbol::intern("list_to_tuple"), 1));
        bifs.insert((Symbol::intern("load_module"), 2));
        bifs.insert((Symbol::intern("make_ref"), 0));
        bifs.insert((Symbol::intern("map_size"),1));
        bifs.insert((Symbol::intern("map_get"),2));
        bifs.insert((Symbol::intern("max"),2));
        bifs.insert((Symbol::intern("min"),2));
        bifs.insert((Symbol::intern("module_loaded"), 1));
        bifs.insert((Symbol::intern("monitor"), 2));
        bifs.insert((Symbol::intern("monitor"), 3));
        bifs.insert((Symbol::intern("monitor_node"), 2));
        bifs.insert((Symbol::intern("node"), 0));
        bifs.insert((Symbol::intern("node"), 1));
        bifs.insert((Symbol::intern("nodes"), 0));
        bifs.insert((Symbol::intern("nodes"), 1));
        bifs.insert((Symbol::intern("now"), 0));
        bifs.insert((Symbol::intern("open_port"), 2));
        bifs.insert((Symbol::intern("pid_to_list"), 1));
        bifs.insert((Symbol::intern("port_to_list"), 1));
        bifs.insert((Symbol::intern("port_close"), 1));
        bifs.insert((Symbol::intern("port_command"), 2));
        bifs.insert((Symbol::intern("port_command"), 3));
        bifs.insert((Symbol::intern("port_connect"), 2));
        bifs.insert((Symbol::intern("port_control"), 3));
        bifs.insert((Symbol::intern("pre_loaded"), 0));
        bifs.insert((Symbol::intern("process_flag"), 2));
        bifs.insert((Symbol::intern("process_flag"), 3));
        bifs.insert((Symbol::intern("process_info"), 1));
        bifs.insert((Symbol::intern("process_info"), 2));
        bifs.insert((Symbol::intern("processes"), 0));
        bifs.insert((Symbol::intern("purge_module"), 1));
        bifs.insert((Symbol::intern("put"), 2));
        bifs.insert((Symbol::intern("ref_to_list"), 1));
        bifs.insert((Symbol::intern("register"), 2));
        bifs.insert((Symbol::intern("registered"), 0));
        bifs.insert((Symbol::intern("round"), 1));
        bifs.insert((Symbol::intern("self"), 0));
        bifs.insert((Symbol::intern("setelement"), 3));
        bifs.insert((Symbol::intern("size"), 1));
        bifs.insert((Symbol::intern("spawn"), 1));
        bifs.insert((Symbol::intern("spawn"), 2));
        bifs.insert((Symbol::intern("spawn"), 3));
        bifs.insert((Symbol::intern("spawn"), 4));
        bifs.insert((Symbol::intern("spawn_link"), 1));
        bifs.insert((Symbol::intern("spawn_link"), 2));
        bifs.insert((Symbol::intern("spawn_link"), 3));
        bifs.insert((Symbol::intern("spawn_link"), 4));
        bifs.insert((Symbol::intern("spawn_monitor"), 1));
        bifs.insert((Symbol::intern("spawn_monitor"), 3));
        bifs.insert((Symbol::intern("spawn_opt"), 2));
        bifs.insert((Symbol::intern("spawn_opt"), 3));
        bifs.insert((Symbol::intern("spawn_opt"), 4));
        bifs.insert((Symbol::intern("spawn_opt"), 5));
        bifs.insert((Symbol::intern("split_binary"), 2));
        bifs.insert((Symbol::intern("statistics"), 1));
        bifs.insert((Symbol::intern("term_to_binary"), 1));
        bifs.insert((Symbol::intern("term_to_binary"), 2));
        bifs.insert((Symbol::intern("throw"), 1));
        bifs.insert((Symbol::intern("time"), 0));
        bifs.insert((Symbol::intern("tl"), 1));
        bifs.insert((Symbol::intern("trunc"), 1));
        bifs.insert((Symbol::intern("tuple_size"), 1));
        bifs.insert((Symbol::intern("tuple_to_list"), 1));
        bifs.insert((Symbol::intern("unlink"), 1));
        bifs.insert((Symbol::intern("unregister"), 1));
        bifs.insert((Symbol::intern("whereis"), 1));

        bifs
    }
}

lazy_static! {
    pub static ref GUARDS: HashSet<(Symbol, usize)> = {
        let guards = HashSet::new();
        guards.insert((Symbol::intern("abs"), 1));
        guards.insert((Symbol::intern("binary_part"), 2));
        guards.insert((Symbol::intern("binary_part"), 3));
        guards.insert((Symbol::intern("bit_size"), 1));
        guards.insert((Symbol::intern("byte_size"), 1));
        guards.insert((Symbol::intern("ceil"), 1));
        guards.insert((Symbol::intern("element"), 2));
        guards.insert((Symbol::intern("float"), 1));
        guards.insert((Symbol::intern("floor"), 1));
        guards.insert((Symbol::intern("hd"), 1));
        guards.insert((Symbol::intern("is_map_key"), 2));
        guards.insert((Symbol::intern("length"), 1));
        guards.insert((Symbol::intern("map_size"), 1));
        guards.insert((Symbol::intern("map_get"), 2));
        guards.insert((Symbol::intern("node"), 0));
        guards.insert((Symbol::intern("node"), 1));
        guards.insert((Symbol::intern("round"), 1));
        guards.insert((Symbol::intern("self"), 0));
        guards.insert((Symbol::intern("size"), 1));
        guards.insert((Symbol::intern("tl"), 1));
        guards.insert((Symbol::intern("trunc"), 1));
        guards.insert((Symbol::intern("tuple_size"), 1));

        guards
    }
}

lazy_static! {
    pub static ref TYPE_TESTS: HashSet<(Symbol, usize)> = {
        let guards = HashSet::new();
        guards.insert((Symbol::intern("is_atom"), 1));
        guards.insert((Symbol::intern("is_binary"), 1));
        guards.insert((Symbol::intern("is_bitstring"), 1));
        guards.insert((Symbol::intern("is_boolean"), 1));
        guards.insert((Symbol::intern("is_float"), 1));
        guards.insert((Symbol::intern("is_function"), 1));
        guards.insert((Symbol::intern("is_function"), 2));
        guards.insert((Symbol::intern("is_integer"), 1));
        guards.insert((Symbol::intern("is_list"), 1));
        guards.insert((Symbol::intern("is_map"), 1));
        guards.insert((Symbol::intern("is_number"), 1));
        guards.insert((Symbol::intern("is_pid"), 1));
        guards.insert((Symbol::intern("is_port"), 1));
        guards.insert((Symbol::intern("is_record"), 2));
        guards.insert((Symbol::intern("is_record"), 3));
        guards.insert((Symbol::intern("is_reference"), 1));
        guards.insert((Symbol::intern("is_tuple"), 1));

        guards
    }
}

pub struct Environment {
    pub defined_modules: HashSet<Symbol>,
    pub deprecated_modules: HashSet<Symbol>,
    pub deprecations: HashMap<ResolvedFunctionName, DeprecatedFlag>,
    pub calls: HashSet<ResolvedFunctionName>,
    pub imported: HashSet<ResolvedFunctionName>,
}
impl Environment {
    pub fn check_deprecated(&self, fname: &ResolvedFunctionName) -> Option<DeprecatedFlag> {
        let module = fname.module.as_ref().unwrap();
        if let Some(flag) = self.deprecated_modules.get(&module.name) {
            return Some(flag);
        }
        match self.deprecations.get(fname) {
            None => None,
            Some(flag) => Some(flag),
        }
    }
    pub fn is_guard(&self, fname: &ResolvedFunctionName) -> bool {
        let key = (fname.function.name.clone(), fname.arity.unwrap());
        GUARDS.contains(&key) || TYPE_TESTS.contains(&key)
    }

    pub fn is_bif(&self, fname: &ResolvedFunctionName) -> bool {
        let key = (fname.function.name.clone(), fname.arity.unwrap());
        BIFS.contains(&key)
    }

    pub fn is_shadowing_bif(&self, fname: &ResolvedFunctionName) -> bool {
        if !self.is_bif(fname) {
            return false;
        }

        // This function conflicts with an auto-imported BIF,
        // but first check whether the auto-import of that BIF was disabled
        if self.no_auto_imports.contains(fname) {
            return false;
        }

        // This function conflicts with an auto-imported BIF
        return true;
    }
}
