// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
        %res = cir.foo %0 : i32
        return
    }
}
