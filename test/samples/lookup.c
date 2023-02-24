#include <string.h>

typedef void (*Fun)();

typedef struct {
    const char *function;
    unsigned arity;
    const Fun fun;
} Function;

typedef struct {
    const char *module;
    const Function *table;
    unsigned funs;
} Module;

void foo_main();
unsigned foo_init(const char **argv, int argc);

Function __LUMEN_MODULE_foo[] = {
    { .function = "main", .arity = 0, .fun = (Fun)foo_main },
    { .function = "init", .arity = 2, .fun = (Fun)foo_init },
};
Module __LUMEN_MODULES[] = {
    { .module = "foo", .table = &__LUMEN_MODULE_foo[0], .funs = 2 },
};
unsigned __LUMEN_MODULES_LEN = 1;

unsigned lookup(const char*, const char*, unsigned);

int main() {
    char *query_module = "foo";
    char *query_fun = "init";
    unsigned query_arity = 2;

    return lookup(query_module, query_fun, query_arity);
}

unsigned lookup(const char *query_module, const char *query_fun, unsigned query_arity) {
    for (unsigned m = 0; m < __LUMEN_MODULES_LEN; m++) {
        Module *module = &__LUMEN_MODULES[m];
        if (strcmp(module->module, query_module) == 0) {
            for (unsigned f = 0; f < module->funs; f++) {
                const Function *function = &module->table[f];
                if (strcmp(function->function, query_fun) == 0) {
                    if (function->arity == query_arity) {
                        return 0;
                    } else {
                        // Invalid arity
                        return 3;
                    }
                }
            }
            // Couldn't find function
            return 2;
        }
    }
    // Couldn't find module
    return 1;
}

void foo_main() {
    return;
}

unsigned foo_init(const char **argv, int argc) {
    return 1;
}

