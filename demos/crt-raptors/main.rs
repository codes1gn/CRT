extern crate backend_vulkan as vk_types;
extern crate chopper_runtime;
extern crate float_eq;
extern crate hal;

use float_eq::{assert_float_eq, float_eq};

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use hal::prelude::*;
use hal::{adapter::Adapter, adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso};

use chopper_runtime::prelude::*;

fn test_pressure() {
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
    assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![8.7], rmax_all <= 0.00001);

    let status = ipt.run_bytecode_eagerly("%0 = crt.literal.const.f32! 1.3 : f32\n");
    let status = ipt.run_bytecode_eagerly("%1 = crt.add.f32! %0, %1 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![10.], rmax_all <= 0.00001);

    let start = Instant::now();
    for k in 1..5000 {
        let status = ipt.run_bytecode_eagerly("%0 = crt.literal.const.f32! 1.3 : f32\n");
        let status = ipt.run_bytecode_eagerly("%1 = crt.add.f32! %0, %1 : f32\n");
        if k % 100 == 0 {
            println!("step {}", k);
        }
    }
    let duration = start.elapsed();
    // assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![1308.7039], rmax_all <= 0.00001);
    println!("time-cost >>>>>>>>>>> {:?}", duration);
}

fn test_mock_bytecode_f32_binary_add_then_sub_f32() {
    let mut ipt = Interpreter::new();
    // ok
    let status = ipt.run_bytecode_eagerly("%8 = crt.literal.const.f32! 1.3 : f32\n");
    let status = ipt.run_bytecode_eagerly("%7 = crt.literal.const.f32! 2.9 : f32\n");
    let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 7.4 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);

    // inspect data valid
    assert_eq!(*ipt.vm.get_raw_vec_f32(8), vec![1.3]);
    assert_eq!(*ipt.vm.get_raw_vec_f32(7), vec![2.9]);

    // add
    let status = ipt.run_bytecode_eagerly("%4 = crt.add.f32! %8, %7 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(*ipt.vm.get_raw_vec_f32(4), vec![4.2], rmax_all <= 0.00001);

    // sub
    let status = ipt.run_bytecode_eagerly("%5 = crt.sub.f32! %1, %4 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    // TODO package this assert macro into utils, hide rmax_all setting from hardcode
    assert_float_eq!(*ipt.vm.get_raw_vec_f32(5), vec![3.2], rmax_all <= 0.00001);
}

fn test_mock_run() {
    let mut ipt = Interpreter::new();
    let status = ipt.run_bytecode_eagerly(
        "%9 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>\n",
    );
    let status = ipt.run_bytecode_eagerly(
        "%7 = crt.literal.const.tensor! dense<[2.2 3.3 3.3 1.1 3.3 2.2], shape=[2 3]>\n",
    );
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);

    // inspect data valid
    assert_float_eq!(
        *ipt.vm.get_raw_vec_f32(9),
        vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        rmax_all <= 0.00001
    );
    assert_eq!(*ipt.vm.get_tensor_shape(9), vec![2, 3]);
    assert_float_eq!(
        *ipt.vm.get_raw_vec_f32(7),
        vec![2.2, 3.3, 3.3, 1.1, 3.3, 2.2],
        rmax_all <= 0.00001
    );
    assert_eq!(*ipt.vm.get_tensor_shape(7), vec![2, 3]);

    // sub
    let status = ipt.run_bytecode_eagerly("%5 = crt.sub.f32! %7, %9 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(
        *ipt.vm.get_raw_vec_f32(5),
        vec![1.1, 1.1, 0.0, -3.3, -2.2, -4.4],
        rmax_all <= 0.00001
    );
}

fn test_bytecode_run() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();

    let data0 = vec![1.1, 2.2, 3.3];
    let data1 = vec![1.1, 2.2, 3.3];
    ipt.vm.push_shaped_tensor_at_pos(0, data0, vec![1, 3]);
    ipt.vm.push_shaped_tensor_at_pos(1, data1, vec![1, 3]);

    ipt.run_bytecode_eagerly("%4 = crt.add.f32! %1, %0 : f32\n");
    assert_float_eq!(
        *ipt.vm.get_raw_vec_f32(4),
        vec![2.2, 4.4, 6.6],
        rmax_all <= 0.00001
    );
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::try_init().unwrap();
    // test_mock_run();
    // test_bytecode_run();
    // test_mock_bytecode_f32_binary_add_then_sub_f32();
    test_pressure();
}
