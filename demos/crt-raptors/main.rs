extern crate backend_vulkan as vk_types;
extern crate chopper_runtime;
extern crate hal;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use hal::prelude::*;
use hal::{adapter::Adapter, adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso};

use chopper_runtime::prelude::*;

fn main() {
    // step 1, init device instance, also in VM instance init part
    let ist = DeviceInstance::new();
    let mut ipt = Interpreter::new(&ist);

    let data0 = vec![1.1, 2.2, 3.3];
    let data1 = vec![1.1, 2.2, 3.3];
    ipt.vm.push_tensor_buffer(0, data0, vec![1, 3]);
    ipt.vm.push_tensor_buffer(1, data1, vec![1, 3]);

    ipt.run_bytecode("%4 = crt.add.f32! %1, %0 : f32\n".to_string());
    let outs = ipt.vm.data_buffer_f32.remove(&4).unwrap();
    println!("{:#?}", outs);
}
