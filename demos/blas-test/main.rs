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
    let status = ipt.mock_operation("%0 = crt.literal.const.f32! 1.3 : f32\n");
    let status = ipt.mock_operation("%1 = crt.literal.const.f32! 7.4 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);

    // add
    let status = ipt.mock_operation("%1 = crt.add.f32! %0, %1 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(*ipt.vm.get_fdata(1), vec![8.7], rmax_all <= 0.00001);

    let status = ipt.mock_operation("%0 = crt.literal.const.f32! 1.3 : f32\n");
    let status = ipt.mock_operation("%1 = crt.add.f32! %0, %1 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(*ipt.vm.get_fdata(1), vec![10.], rmax_all <= 0.00001);

    let start = Instant::now();
    for k in 1..1000 {
        let status = ipt.mock_operation("%0 = crt.literal.const.f32! 1.3 : f32\n");
        let status = ipt.mock_operation("%1 = crt.add.f32! %0, %1 : f32\n");
        if k % 100 == 0 {
            println!("step {}", k);
        }
    }
    let duration = start.elapsed();
    assert_float_eq!(*ipt.vm.get_fdata(1), vec![1308.7039], rmax_all <= 0.00001);
    println!("time-cost >>>>>>>>>>> {:?}", duration);
}

fn add_test() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new();
    ipt.init(3);

    let data0 = vec![1.1, 2.2, 3.3];
    let data1 = vec![1.1, 2.2, 3.3];
    ipt.vm.push_tensor_buffer(0, data0, vec![1, 3]);
    ipt.vm.push_tensor_buffer(1, data1, vec![1, 3]);

    ipt.run_bytecode("%4 = crt.add.f32! %1, %0 : f32\n".to_string());
    assert_float_eq!(
        *ipt.vm.get_fdata(4),
        vec![2.2, 4.4, 6.6],
        rmax_all <= 0.00001
    );
}

fn main() {
    // add_test();
    pressure_test();
}
