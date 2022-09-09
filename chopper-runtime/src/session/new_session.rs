extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, env, fs, iter, path::Path, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};
use raptors::prelude::*;

use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_view::*;
use crate::device_context::new_device_context::NewDeviceContext;
use crate::device_context::*;
use crate::functor::new_functor::NewFunctor;
use crate::functor::*;
use crate::instance::*;
use crate::instruction;

// make it pub(crate) -> pub
#[derive(Debug)]
pub struct NewSession {
    pub(crate) device_context: NewDeviceContext,
    pub actor_system: ActorSystemHandle,
}

impl Drop for NewSession {
    fn drop(&mut self) {
        unsafe {
            println!("drop NewSession");
        };
    }
}

impl NewSession {
    #[tokio::main]
    pub async fn new() -> NewSession {
        let mut device_context = NewDeviceContext::new();
        let mut system = build_system!("Raptors");

        return Self {
            device_context: device_context,
            actor_system: system,
        };
    }

    pub fn init(&mut self) {
        // TODO support more kernels
        // TODO get rid of path hardcode by cargo manage datafiles of kernels
        // let kernel_path = vec![kernel::KERNELPATH];
        let path = env::current_dir().unwrap();
        // println!("{}", path.display());

        self.device_context.register_kernels(
            "/root/project/glsl_src/binary_arithmetic_f32.comp",
            String::from("binary_arithmetic_f32"),
        );
        self.device_context.register_kernels(
            "/root/project/glsl_src/binary_arithmetic_i32.comp",
            String::from("binary_arithmetic_i32"),
        );
        self.device_context.register_kernels(
            "/root/project/glsl_src/matrix_multiple_f32.comp",
            //    "/root/project/chopper/backend-rs/chopper-runtime/src/kernel/glsl_src/matrix_multiple_f32.comp",
            String::from("matrix_multiple_f32"),
        );
    }

    pub fn benchmark_run<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        opcode: instruction::OpCode,
        lhs_tensor: TensorView<T>,
        rhs_tensor: TensorView<T>,
        // TODO-trial lowering UniBuffer range, to make session dev independent
        // lhs_dataview: UniBuffer<concrete_backend::Backend, T>,
        // rhs_dataview: UniBuffer<concrete_backend::Backend, T>,
        // ) -> UniBuffer<concrete_backend::Backend, T> {
    ) -> TensorView<T> {
        self.device_context.device.start_capture();
        let outs = self.run::<T>(opcode, lhs_tensor, rhs_tensor);
        self.device_context.device.stop_capture();
        outs
    }

    pub fn run<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        opcode: instruction::OpCode,
        lhs_tensor: TensorView<T>,
        rhs_tensor: TensorView<T>,
    ) -> TensorView<T> {
        // step 2 open a physical compute device

        // step 4 load compiled spirv
        //
        let mut out_tensor = self.device_context.compute(lhs_tensor, rhs_tensor, opcode);

        // let mut result_buffer =
        //    NewFunctor::new().apply::<T>(&mut self.device_context, lhs_dataview, rhs_dataview, opcode);

        // clear shader module
        // self.device_context.device.destroy_shader_module(shader);

        // update dataview with new value
        // result_buffer.eval(&self.device_context.device);
        // print outs
        out_tensor
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_create_session() {
        // defaultly to Add, TODO, add more dispatch path
        let session = NewSession::new();
        assert_eq!(0, 0);
    }

    #[test]
    fn test_e2e_add() {
        let mut se = NewSession::new();
        se.init();
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![11.0, 13.0, 17.0];
        let lhs_shape = vec![lhs.len()];
        let rhs_shape = vec![lhs.len()];
        // create lhs dataview
        let lhs_tensor_view = TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape);
        let rhs_tensor_view = TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape);
        // TODO-trial lowering UniBuffer range, to make session dev independent
        // let mut lhs_dataview = UniBuffer::<concrete_backend::Backend, f32>::new(
        //     &se.device_context.device,
        //     &se.device_context
        //         .device_instance
        //         .memory_property()
        //         .memory_types,
        //     lhs_tensor_view,
        // );
        // let mut rhs_dataview = UniBuffer::<concrete_backend::Backend, f32>::new(
        //     &se.device_context.device,
        //     &se.device_context
        //         .device_instance
        //         .memory_property()
        //         .memory_types,
        //     rhs_tensor_view,
        // );
        let opcode = instruction::OpCode::ADDF32;
        let mut result_buffer = se.benchmark_run(opcode, lhs_tensor_view, rhs_tensor_view);
        // let mut result_buffer = se.benchmark_run(opcode, lhs_dataview, rhs_dataview);
        assert_eq!(result_buffer.data, vec!(12.0, 15.0, 20.0));
    }

    #[test]
    fn test_e2e_sub() {
        let mut se = NewSession::new();
        se.init();
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![11.0, 13.0, 17.0];
        let lhs_shape = vec![lhs.len()];
        let rhs_shape = vec![lhs.len()];
        // create lhs dataview
        let lhs_tensor_view = TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape);
        let rhs_tensor_view = TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape);
        let opcode = instruction::OpCode::SUBF32;
        let mut result_buffer = se.benchmark_run(opcode, lhs_tensor_view, rhs_tensor_view);
        assert_eq!(result_buffer.data, vec!(-10.0, -11.0, -14.0));
    }

    #[test]
    fn test_e2e_matmul() {
        let mut se = NewSession::new();
        se.init();
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let lhs_shape = vec![2, 3];
        let rhs_shape = vec![3, 2];
        //create lhs dataview
        let lhs_tensor_view = TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape);
        let rhs_tensor_view = TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape);
        let opcode = instruction::OpCode::MATMULF32;
        let mut result_buffer = se.benchmark_run(opcode, lhs_tensor_view, rhs_tensor_view);
    }
}
