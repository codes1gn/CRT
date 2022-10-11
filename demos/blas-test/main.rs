extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

fn pressure_test() {
    let mut ipt = Interpreter::new();
    ipt.init(1);
    // ok
    let status = ipt.run_bytecode_eagerly("%0 = crt.literal.const.f32! 1.3 : f32\n");
    let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 7.4 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);

    // add
    let status = ipt.run_bytecode_eagerly("%1 = crt.add.f32! %0, %1 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(*ipt.vm.get_fdata(1), vec![8.7], rmax_all <= 0.00001);

    let status = ipt.run_bytecode_eagerly("%0 = crt.literal.const.f32! 1.3 : f32\n");
    let status = ipt.run_bytecode_eagerly("%1 = crt.add.f32! %0, %1 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(*ipt.vm.get_fdata(1), vec![10.], rmax_all <= 0.00001);

    let start = Instant::now();
    for k in 1..1000 {
        let status = ipt.run_bytecode_eagerly("%0 = crt.literal.const.f32! 1.3 : f32\n");
        let status = ipt.run_bytecode_eagerly("%1 = crt.add.f32! %0, %1 : f32\n");
        if k % 100 == 0 {
            println!("step {}", k);
        }
    }
    let duration = start.elapsed();
    assert_float_eq!(*ipt.vm.get_fdata(1), vec![1308.7039], rmax_all <= 0.00001);
    println!("time-cost >>>>>>>>>>> {:?}", duration);
}

fn big_add_test() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();
    ipt.init(3);

    ipt.run_bytecode_eagerly("%0 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");
    ipt.run_bytecode_eagerly("%1 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");

    // TODO svalue<[shape], 0.7>
    // ipt.run_bytecode_eagerly("%4 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");

    ipt.run_bytecode_eagerly("%3 = crt.add.f32! %1, %0 : f32\n");
    assert_float_eq!(
        *ipt.vm.get_fdata(3),
        vec![2.0; 34 * 82 * 3],
        rmax_all <= 0.00001
    );
}

fn exp_test() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();
    ipt.init(3);

    ipt.run_bytecode_eagerly("%0 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");

    // TODO svalue<[shape], 0.7>
    // ipt.run_bytecode_eagerly("%4 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");

    ipt.run_bytecode_eagerly("%1 = crt.exp.f32! %0 : f32\n");
    // ipt.run_bytecode_eagerly("%2 = crt.exp.f32! %1 : f32\n");
    // ipt.run_bytecode_eagerly("%3 = crt.exp.f32! %2 : f32\n");
    // assert_float_eq!(
    //     *ipt.vm.get_fdata(3),
    //     vec![1.0; 34 * 82 * 3],
    //     rmax_all <= 0.00001
    // );
}

fn small_add_test() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();
    ipt.init(1);

    let data0 = vec![1.1, 2.2, 3.3];
    let data1 = vec![1.1, 2.2, 3.3];
    ipt.vm.push_tensor_buffer(0, data0, vec![1, 3]);
    ipt.vm.push_tensor_buffer(1, data1, vec![1, 3]);

    ipt.run_bytecode_eagerly("%4 = crt.add.f32! %1, %0 : f32\n");
    assert_float_eq!(
        *ipt.vm.get_fdata(4),
        vec![2.2, 4.4, 6.6],
        rmax_all <= 0.00001
    );
}

fn main() {
    // small_add_test();
    // big_add_test();
    // pressure_test();
    exp_test();
}
