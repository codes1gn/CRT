extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

fn sequence_test() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();
    ipt.init(3);

    // TODO svalue<[shape], 0.7>
    // ipt.run_bytecode("%0 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n");
    // ipt.run_bytecode("%1 = crt.exp.f32! %0 : f32\n");
    // ipt.run_bytecode("%2 = crt.exp.f32! %1 : f32\n");
    // ipt.run_bytecode("%3 = crt.exp.f32! %2 : f32\n");
    // ipt.run_bytecode("%4 = crt.exp.f32! %3 : f32\n");

    // bytecode that combines above operations in one string
    let bytecode = "
        %0 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n\
        %1 = crt.exp.f32! %0 : f32\n\
        %2 = crt.exp.f32! %1 : f32\n\
        %3 = crt.exp.f32! %2 : f32\n\
        %11 = crt.exp.f32! %0 : f32\n\
        %21 = crt.exp.f32! %1 : f32\n\
        %33 = crt.exp.f32! %2 : f32\n\
        return %33\n
    ";
    // ipt.run_bytecode_eagerly(bytecode);
    ipt.run_bytecode_lazily(bytecode);
    assert_float_eq!(*ipt.vm.get_fdata(33), vec![1.0; 4], rmax_all <= 0.00001);
}

fn main() {
    sequence_test();
}
