extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, collections::HashMap, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use raptors::prelude::*;

use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_view::*;
use crate::functor::TensorFunctor;
use crate::functor::*;
use crate::instance::*;
use crate::instruction;
use crate::instruction::*;
use crate::kernel::kernel_registry::KernelRegistry;

#[derive(Debug)]
pub struct DeviceContext {
    // TODO refactor into kernel_registry
    pub kernel_registry: KernelRegistry,
    //adapter: Adapter<concrete_backend::Backend>,
    //physical_device: concrete_backend::PhysicalDevice,
    //pub device_and_queue: hal::adapter::Gpu<concrete_backend::Backend>,
    // TODO .first_mut().unwrap(); before use, owner is Functor
    //
    pub descriptor_pool: concrete_backend::native::DescriptorPool,
    pub queue_groups: Vec<hal::queue::family::QueueGroup<concrete_backend::Backend>>,
    pub device: concrete_backend::Device,

    // FIX: device instance have to put at last since the drop rule of Rust is a sequence order
    // rather than a reverse order in struct. Thus, we have to make sure device instance is dropped
    // at last
    pub device_instance: DeviceInstance,
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            // self.device.destroy_descriptor_pool(self.descriptor_pool);
            println!("drop::DeviceContext");
        };
    }
}

// kaigao
impl ExecutorLike for DeviceContext {
    type TensorType = TensorView<f32>;
    fn new() -> DeviceContext {
        let mut di = DeviceInstance::new();
        let mut device_and_queue = di.device_and_queue();
        let mut descriptor_pool = unsafe {
            device_and_queue.device.create_descriptor_pool(
                100, // TODO count of desc sets which below max_sets
                iter::once(pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                }),
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create descriptor pool");
        return Self {
            device_instance: di,
            kernel_registry: KernelRegistry::new(),
            //device_and_queue: device_and_queue,
            device: device_and_queue.device,
            queue_groups: device_and_queue.queue_groups,
            descriptor_pool: descriptor_pool,
        };
    }
    fn compute(&self, wkl: Self::TensorType) -> Self::TensorType {
        println!("============ on computing =============");
        wkl
    }
}

impl DeviceContext {
    pub fn register_kernels(&mut self, file_path: &str, query_entry: String) {
        // glsl_to_spirv, TODO, support more, spv format and readable spirv ir.
        // TODO, read external config of all kernels, and cache it by OpCode
        let glsl = fs::read_to_string(file_path).unwrap();
        // println!("{:?}", glsl);
        let spirv_file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Compute).unwrap();
        // println!("{:?}", spirv_file);
        // TODO need to impl implicit conversion
        let spirv: KernelByteCode = auxil::read_spirv(spirv_file).unwrap() as KernelByteCode;
        //let spirv: KernelByteCode = auxil::read_spirv(spirv_file).unwrap() as KernelByteCode;
        self.kernel_registry.register(spirv, query_entry);
    }

    pub fn kernel_registry(&self) -> &KernelRegistry {
        &self.kernel_registry
    }

    // TODO seal raptors OpCode inside and not expose
    pub fn dispatch_kernel(&self, op: instruction::OpCode) -> Kernel {
        let query_entry: String = op.to_kernel_query_entry();
        self.kernel_registry.dispatch_kernel(self, op, query_entry)
    }

    pub fn compute_legacy<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        lhs_tensor: TensorView<T>,
        rhs_tensor: TensorView<T>,
        opcode: instruction::OpCode,
    ) -> TensorView<T> {
        let mut lhs_buffer_functor = UniBuffer::<concrete_backend::Backend, T>::new(
            &self.device,
            &self.device_instance.memory_property().memory_types,
            lhs_tensor,
        );
        let mut rhs_buffer_functor = UniBuffer::<concrete_backend::Backend, T>::new(
            &self.device,
            &self.device_instance.memory_property().memory_types,
            rhs_tensor,
        );
        let mut out_buffer_functor =
            TensorFunctor::new().apply::<T>(self, lhs_buffer_functor, rhs_buffer_functor, opcode);
        // TODO-fix destroy memory when compute done, consider keep this in future for fusion
        // purpose
        out_buffer_functor.try_drop(&self.device);

        // consume this UniBuffer and wrap a tensorview, before drop UniBuffer, destroy the real
        // memory
        let out_tensor = TensorView::<T>::new(
            out_buffer_functor.raw_data,
            ElementType::F32,
            out_buffer_functor.shape,
        );
        out_tensor
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn create_add_functor() {
        // defaultly to Add, TODO, add more dispatch path
        let add_functor = DeviceContext::new();
        assert_eq!(0, 0);
    }
}
