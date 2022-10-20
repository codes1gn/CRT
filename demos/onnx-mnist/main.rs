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
    let bytecode = "
        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32\n\
        %1 = crt.reshape! %0, [4096 1024]\n\
        %2 = crt.helper.svalue.tensor! ones<[1024 1024]>: f32\n\
        %3 = crt.matmul.f32! %1, %2 : f32\n\
        %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32\n\
        %5 = crt.add.f32! %3, %4: f32\n\
        return %5\n
    ";
    ipt.run_bytecode_lazily(bytecode);
    ipt.vm.dump_tensor_f32(5);
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::try_init().unwrap();
    bert_block();
}
