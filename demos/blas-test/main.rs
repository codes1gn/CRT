extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

fn add_test() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();

    // let data0 = vec![1.1, 2.2, 3.3];
    // let data1 = vec![1.1, 2.2, 3.3];
    // ipt.vm.push_tensor_buffer(0, data0, vec![1, 3]);
    // ipt.vm.push_tensor_buffer(1, data1, vec![1, 3]);

    // ipt.run_bytecode("%4 = crt.add.f32! %1, %0 : f32\n".to_string());
    // assert_float_eq!(
    //     *ipt.vm.get_fdata(4),
    //     vec![2.2, 4.4, 6.6],
    //     rmax_all <= 0.00001
    // );
}

fn main() {
    add_test();
}
