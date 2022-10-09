use std::collections::HashMap;

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::base::kernel::*;
use crate::instruction::*;
use crate::vkgpu_executor::*;

// TODO move all vulkan features into vulkan_device folder
// then simplify this feature config with lib.rs
#[derive(Debug)]
pub struct KernelRegistry {
    pub executable_cache_table: HashMap<String, KernelByteCode>,
}

impl Drop for KernelRegistry {
    fn drop(&mut self) {
        unsafe {
            println!("drop::KernelRegistry");
        };
    }
}

impl KernelRegistry {
    pub fn new() -> KernelRegistry {
        return Self {
            executable_cache_table: HashMap::new(),
        };
    }

    fn query_kernel_cache(&self, opcode: CRTOpCode, query_entry: String) -> &KernelByteCode {
        // TODO dummy impl
        return self.executable_cache_table.get(&query_entry).unwrap();
    }

    pub fn register(&mut self, kernel: KernelByteCode, query_entry: String) {
        self.executable_cache_table.insert(query_entry, kernel);
    }

    pub fn dispatch_kernel(
        &self,
        dc: &VkGPUExecutor,
        op: CRTOpCode,
        query_entry: String,
    ) -> Kernel {
        let shader = unsafe {
            dc.device
                .create_shader_module(self.query_kernel_cache(op, query_entry))
        }
        .unwrap();
        shader
    }
}
