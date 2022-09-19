extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, env, fs, iter, path::Path, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};
use raptors::prelude::*;

use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_view::*;
use crate::device_context::*;
use crate::functor::TensorFunctor;
use crate::functor::*;
use crate::instance::*;
use crate::instruction;

// TODO-move to interpreter initialization
use opentelemetry::global;
use tokio::io::Result;
// TODO(avoid bug): use tracing::info;
use log::{debug, info};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

// make it pub(crate) -> pub
#[derive(Debug)]
pub struct HostSession {
    pub(crate) device_context: DeviceContext,
    pub actor_system: ActorSystemHandle<DeviceContext, TensorView<f32>>,
}

impl Drop for HostSession {
    fn drop(&mut self) {
        unsafe {
            println!("drop HostSession");
        };
    }
}

#[macro_export]
macro_rules! build_crt {
    ($name:expr) => {{
        let mut sys_config = SystemConfig::new($name, "info");
        let mut sys_builder = SystemBuilder::new();
        sys_config.set_ranks(0 as usize);
        let system = sys_builder.build_with_config::<DeviceContext, TensorView<f32>>(sys_config);
        system
    }};
}

impl HostSession {
    #[tokio::main]
    pub async fn new() -> HostSession {
        let mut device_context = DeviceContext::new();

        // init raptors env
        // TODO(debug) fix to allow tracing with vk_device, perhaps backend vulkan uses env_logger
        // this causes conflict
        std::env::set_var("RUST_LOG", "info");
        // if std::env::args().any(|arg| arg == "--trace") {
        //     global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
        //     let tracer = opentelemetry_jaeger::new_pipeline()
        //         .with_service_name("raptors")
        //         .install_simple()
        //         .unwrap();

        //     let opentelemetry = tracing_opentelemetry::layer().with_tracer(tracer);
        //     tracing_subscriber::registry()
        //         .with(opentelemetry)
        //         .with(fmt::Layer::default())
        //         .try_init()
        //         .unwrap();
        // } else {
        //     //tracing_subscriber::fmt::try_init().unwrap();
        //     env_logger::init();
        // };

        // perform raptors actions
        let mut system = build_crt!("Raptors");
        // let system = {
        //     let mut sys_config = SystemConfig::new($name, "info");
        //     let mut sys_builder = SystemBuilder::new();
        //     sys_config.set_ranks(0 as usize);
        //     let system = sys_builder.build_with_config::<Executor, Workload>(sys_config);
        //     system
        // };
        //
        // spawn actors, in the dev, we try with two actors
        // one with vk device context that owns the gpu
        // the other works with CPU
        // TODO: add a CPU high perf lib, maybe rayon
        // TODO: maybe remove Clone requirements for type parameter U = TensorView<f32>
        let msg: TypedMessage<TensorView<f32>> = build_msg!("spawn", 1);
        system.issue_order(msg).await;

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
        let opmsg = match opcode {
            instruction::OpCode::ADDF32 => build_comp_msg!("add-op"),
            instruction::OpCode::SUBF32 => build_comp_msg!("sub-op"),
            _ => build_msg!("identity-op"),
        };
        println!("{:#?}", opmsg);
        // self.actor_system.issue_order(opmsg.clone()).await;

        let mut out_tensor = self
            .device_context
            .compute_legacy(lhs_tensor, rhs_tensor, opcode);

        // let mut result_buffer =
        //    TensorFunctor::new().apply::<T>(&mut self.device_context, lhs_dataview, rhs_dataview, opcode);

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
        let session = HostSession::new();
        assert_eq!(0, 0);
    }

    #[test]
    fn test_e2e_add() {
        let mut se = HostSession::new();
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
        let mut se = HostSession::new();
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
        let mut se = HostSession::new();
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
