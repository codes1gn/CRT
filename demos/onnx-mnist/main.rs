extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

// %0 = crt.helper.svalue [32 128 1024]: f32
// %1 = crt.reshape %0 [4096 1024]: f32
//
fn bert_block() {
    let mut ipt = Interpreter::new();
    ipt.init(3);

    // TODO add support broadcast
    // TODO support comment
    let bytecode = "
        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        %1 = crt.reshape! %0, [4096 1024]

        %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
        %3 = crt.matmul.f32! %1, %2 : f32
        %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
        %5 = crt.add.f32! %3, %4: f32
        %6 = crt.reshape! %5, [32 128 16 64]
        %101 = crt.transpose! %6, [32 16 128 64]

    ";
    // %7 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
    // %8 = crt.matmul.f32! %1, %7 : f32
    // %9 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
    // %10 = crt.add.f32! %8, %9: f32
    // %11 = crt.reshape! %10, [32 128 16 64]
    // %102 = crt.transpose! %11, [32 16 128 64]

    // return %8\n
    ipt.run_bytecode_lazily(bytecode);

    // ipt.vm.dump_tensor_f32(6);
    // ipt.vm.dump_tensor_f32(8);
    // ipt.vm.dump_tensor_f32(101);
    //
    // ipt.vm.dump_tensor_f32(10);
    // ipt.vm.dump_tensor_f32(11);
    // ipt.vm.dump_tensor_f32(102);
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::try_init().unwrap();
    bert_block();
}