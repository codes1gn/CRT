extern crate backend_vulkan as vk_types;
extern crate chopper_runtime;
extern crate float_eq;
extern crate hal;

use float_eq::{assert_float_eq, float_eq};

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use hal::prelude::*;
use hal::{adapter::Adapter, adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso};

use chopper_runtime::prelude::*;

fn test_mock_run() {
    let mut ipt = NewInterpreter::new();
    let status = ipt.mock_operation(
        "%9 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>\n",
    );
    let status = ipt.mock_operation(
        "%7 = crt.literal.const.tensor! dense<[2.2 3.3 3.3 1.1 3.3 2.2], shape=[2 3]>\n",
    );
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);

    // inspect data valid
    assert_float_eq!(
        *ipt.vm.get_fdata(9),
        vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        rmax_all <= 0.00001
    );
    assert_eq!(*ipt.vm.get_fshape(9), vec![2, 3]);
    assert_float_eq!(
        *ipt.vm.get_fdata(7),
        vec![2.2, 3.3, 3.3, 1.1, 3.3, 2.2],
        rmax_all <= 0.00001
    );
    assert_eq!(*ipt.vm.get_fshape(7), vec![2, 3]);

    // sub
    let status = ipt.mock_operation("%5 = crt.sub.f32! %7, %9 : f32\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    assert_float_eq!(
        *ipt.vm.get_fdata(5),
        vec![1.1, 1.1, 0.0, -3.3, -2.2, -4.4],
        rmax_all <= 0.00001
    );
}

fn test_bytecode_run() {
    // step 1, init device instance, also in VM instance init part
    // let ist = DeviceInstance::new();
    let mut ipt = NewInterpreter::new();

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
    test_mock_run();
    test_bytecode_run();
}
