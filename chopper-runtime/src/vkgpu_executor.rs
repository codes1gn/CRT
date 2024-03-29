use std::{borrow::Cow, collections::HashMap, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use raptors::prelude::*;

use crate::instruction;
use crate::instruction::*;

use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_types::*;
use crate::functor::TensorFunctor;
use crate::functor::*;
use crate::instance::*;
use crate::kernel::kernel_registry::KernelRegistry;
use crate::tensors::*;

#[derive(Debug)]
pub struct VkGPUExecutor {
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

impl Drop for VkGPUExecutor {
    fn drop(&mut self) {
        unsafe {
            // self.device.destroy_descriptor_pool(self.descriptor_pool);
            println!("drop::VkGPUExecutor");
        };
    }
}

impl VkGPUExecutor {
    pub fn new() -> VkGPUExecutor {
        println!(" >>> Trying to Init-VkDevice, if panick here, no VkDevice on this machine");
        let mut di = DeviceInstance::new();
        println!("finish to Init-VkDevice");
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
            device: device_and_queue.device,
            queue_groups: device_and_queue.queue_groups,
            descriptor_pool: descriptor_pool,
        };
    }

    pub(crate) fn raw_init(&mut self) {
        self.register_kernels(
            "/root/project/glsl_src/binary_arithmetic_f32.comp",
            String::from("binary_arithmetic_f32"),
        );
        self.register_kernels(
            "/root/project/glsl_src/binary_arithmetic_i32.comp",
            String::from("binary_arithmetic_i32"),
        );
        self.register_kernels(
            "/root/project/glsl_src/matrix_multiple_f32.comp",
            //    "/root/project/chopper/backend-rs/chopper-runtime/src/kernel/glsl_src/matrix_multiple_f32.comp",
            String::from("matrix_multiple_f32"),
        );
    }

    pub(crate) fn binary_compute_i32(
        &mut self,
        op: CRTOpCode,
        lhs_tensor: TensorView<i32>,
        rhs_tensor: TensorView<i32>,
    ) -> TensorView<i32> {
        // println!("============ on computing binary =============");
        // default dtype for compute
        let mut lhs_buffer_functor = UniBuffer::<concrete_backend::Backend, i32>::new(
            &self.device,
            &self.device_instance.memory_property().memory_types,
            lhs_tensor,
        );
        let mut rhs_buffer_functor = UniBuffer::<concrete_backend::Backend, i32>::new(
            &self.device,
            &self.device_instance.memory_property().memory_types,
            rhs_tensor,
        );

        let mut out_buffer_functor =
            TensorFunctor::new().apply::<i32>(self, lhs_buffer_functor, rhs_buffer_functor, op);
        // TODO-fix destroy memory when compute done, consider keep this in future for fusion
        // purpose
        out_buffer_functor.try_drop(&self.device);

        // consume this UniBuffer and wrap a tensorview, before drop UniBuffer, destroy the real
        // memory
        let out_tensor = TensorView::<i32>::new(
            out_buffer_functor.raw_data,
            ElementType::I32,
            out_buffer_functor.shape,
        );
        out_tensor
    }

    pub(crate) fn binary_compute_f32(
        &mut self,
        op: CRTOpCode,
        lhs_tensor: TensorView<f32>,
        rhs_tensor: TensorView<f32>,
    ) -> TensorView<f32> {
        // println!("============ on computing binary =============");
        // default dtype for compute
        let mut lhs_buffer_functor = UniBuffer::<concrete_backend::Backend, f32>::new(
            &self.device,
            &self.device_instance.memory_property().memory_types,
            lhs_tensor,
        );
        let mut rhs_buffer_functor = UniBuffer::<concrete_backend::Backend, f32>::new(
            &self.device,
            &self.device_instance.memory_property().memory_types,
            rhs_tensor,
        );

        let mut out_buffer_functor =
            TensorFunctor::new().apply::<f32>(self, lhs_buffer_functor, rhs_buffer_functor, op);
        // TODO-fix destroy memory when compute done, consider keep this in future for fusion
        // purpose
        out_buffer_functor.try_drop(&self.device);

        // consume this UniBuffer and wrap a tensorview, before drop UniBuffer, destroy the real
        // memory
        let out_tensor = TensorView::<f32>::new(
            out_buffer_functor.raw_data,
            ElementType::F32,
            out_buffer_functor.shape,
        );
        out_tensor
    }

    pub fn register_kernels(&mut self, file_path: &str, query_entry: String) {
        // glsl_to_spirv, TODO, support more, spv format and readable spirv ir.
        // TODO, read external config of all kernels, and cache it by CRTOpCode
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

    // TODO seal raptors CRTOpCode inside and not expose
    pub fn dispatch_kernel(&self, op: CRTOpCode) -> Kernel {
        let query_entry: String = op.to_kernel_query_entry();
        self.kernel_registry.dispatch_kernel(self, op, query_entry)
    }

    pub fn compute_legacy<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        lhs_tensor: TensorView<T>,
        rhs_tensor: TensorView<T>,
        opcode: CRTOpCode,
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

// kaigao
// WIP impl ExecutorLike for VkGPUExecutor {
// WIP     type TensorType = ActTensorTypes;
// WIP     type OpCodeType = CRTOpCode;
// WIP     fn new() -> VkGPUExecutor {
// WIP         let mut di = DeviceInstance::new();
// WIP         let mut device_and_queue = di.device_and_queue();
// WIP         let mut descriptor_pool = unsafe {
// WIP             device_and_queue.device.create_descriptor_pool(
// WIP                 100, // TODO count of desc sets which below max_sets
// WIP                 iter::once(pso::DescriptorRangeDesc {
// WIP                     ty: pso::DescriptorType::Buffer {
// WIP                         ty: pso::BufferDescriptorType::Storage { read_only: false },
// WIP                         format: pso::BufferDescriptorFormat::Structured {
// WIP                             dynamic_offset: false,
// WIP                         },
// WIP                     },
// WIP                     count: 1,
// WIP                 }),
// WIP                 pso::DescriptorPoolCreateFlags::empty(),
// WIP             )
// WIP         }
// WIP         .expect("Can't create descriptor pool");
// WIP         return Self {
// WIP             device_instance: di,
// WIP             kernel_registry: KernelRegistry::new(),
// WIP             //device_and_queue: device_and_queue,
// WIP             device: device_and_queue.device,
// WIP             queue_groups: device_and_queue.queue_groups,
// WIP             descriptor_pool: descriptor_pool,
// WIP         };
// WIP     }
// WIP
// WIP     fn init(&mut self) {
// WIP         // TODO support more kernels
// WIP         // TODO get rid of path hardcode by cargo manage datafiles of kernels
// WIP         // let kernel_path = vec![kernel::KERNELPATH];
// WIP         // let path = env::current_dir().unwrap();
// WIP         // println!("{}", path.display());
// WIP
// WIP         self.register_kernels(
// WIP             "/root/project/glsl_src/binary_arithmetic_f32.comp",
// WIP             String::from("binary_arithmetic_f32"),
// WIP         );
// WIP         self.register_kernels(
// WIP             "/root/project/glsl_src/binary_arithmetic_i32.comp",
// WIP             String::from("binary_arithmetic_i32"),
// WIP         );
// WIP         self.register_kernels(
// WIP             "/root/project/glsl_src/matrix_multiple_f32.comp",
// WIP             //    "/root/project/chopper/backend-rs/chopper-runtime/src/kernel/glsl_src/matrix_multiple_f32.comp",
// WIP             String::from("matrix_multiple_f32"),
// WIP         );
// WIP     }
// WIP
// WIP     fn mock_compute(&mut self, wkl: Self::TensorType) -> Self::TensorType {
// WIP         // println!("============ on computing =============");
// WIP         wkl
// WIP     }
// WIP
// WIP     fn unary_compute(&mut self, op: Self::OpCodeType, lhs: Self::TensorType) -> Self::TensorType {
// WIP         // println!("============ on computing unary =============");
// WIP         lhs
// WIP     }
// WIP
// WIP     fn binary_compute(
// WIP         &mut self,
// WIP         op: Self::OpCodeType,
// WIP         lhs_tensor: Self::TensorType,
// WIP         rhs_tensor: Self::TensorType,
// WIP     ) -> Self::TensorType {
// WIP         // println!("============ on computing binary =============");
// WIP         match lhs_tensor {
// WIP             ActTensorTypes::F32Tensor { data } => {
// WIP                 let lhs_data = data;
// WIP                 match rhs_tensor {
// WIP                     ActTensorTypes::F32Tensor { data } => {
// WIP                         let rhs_data = data;
// WIP                         return ActTensorTypes::F32Tensor {
// WIP                             data: self.binary_compute_f32(op, lhs_data, rhs_data),
// WIP                         };
// WIP                     }
// WIP                     _ => panic!("dtype mismatch"),
// WIP                 }
// WIP             }
// WIP             ActTensorTypes::I32Tensor { data } => {
// WIP                 let lhs_data = data;
// WIP                 match rhs_tensor {
// WIP                     ActTensorTypes::I32Tensor { data } => {
// WIP                         let rhs_data = data;
// WIP                         return ActTensorTypes::I32Tensor {
// WIP                             data: self.binary_compute_i32(op, lhs_data, rhs_data),
// WIP                         };
// WIP                     }
// WIP                     _ => panic!("dtype mismatch"),
// WIP                 }
// WIP             }
// WIP             _ => panic!("dtype-comp not implemented"),
// WIP         };
// WIP     }
// WIP }

#[cfg(test)]

mod tests {
    use super::*;

    #[cfg(feature = "vulkan")]
    #[test]
    fn create_add_functor() {
        // defaultly to Add, TODO, add more dispatch path
        let add_functor = VkGPUExecutor::new();
        assert_eq!(0, 0);
    }
}
