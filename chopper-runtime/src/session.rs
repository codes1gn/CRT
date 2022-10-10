use std::{borrow::Cow, env, fs, iter, path::Path, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};
use raptors::prelude::*;
use tokio::io::Result;
use tokio::sync::oneshot;
use tracing::{debug, info};

use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_types::*;
use crate::executors::*;
use crate::functor::TensorFunctor;
use crate::functor::*;
use crate::instance::*;
use crate::tensors::*;
// use crate::vkgpu_executor::*;
use crate::CRTOpCode;

// make it pub(crate) -> pub
#[derive(Debug)]
pub struct HostSession {
    pub actor_system: ActorSystemHandle<ActExecutorTypes, ActTensorTypes, CRTOpCode>,
    // WIP pub actor_system: ActorSystemHandle<VkGPUExecutor, ActTensorTypes, CRTOpCode>,
    pub async_runtime: tokio::runtime::Runtime,
}

impl Drop for HostSession {
    fn drop(&mut self) {
        unsafe {
            info!("CRT-HostSession dropping");
        };
    }
}

#[macro_export]
macro_rules! build_crt {
    ($name:expr) => {{
        let mut sys_config = SystemConfig::new($name, "info");
        let mut sys_builder = SystemBuilder::new();
        sys_config.set_ranks(0 as usize);
        let system =
            sys_builder.build_with_config::<ActExecutorTypes, ActTensorTypes, CRTOpCode>(sys_config);
        // WIP sys_builder.build_with_config::<VkGPUExecutor, ActTensorTypes, CRTOpCode>(sys_config);
        system
    }};
}

impl HostSession {
    pub fn new() -> HostSession {
        let asrt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let syst = asrt.block_on(async {
            let mut system = build_crt!("Raptors");
            // TODO move this init out of new with init: with config setting (num of each type of
            // devices)
            return system;
        });

        return Self {
            actor_system: syst,
            async_runtime: asrt,
        };
    }

    // TODO refactor this workaround: config
    #[cfg(all(not(feature = "mock"), not(feature = "vulkan"), not(feature = "blas")))]
    pub fn init(&mut self, executor_cnt: usize) {
        panic!("features not set");
    }

    #[cfg(all(feature = "mock", not(feature = "blas"), not(feature = "vulkan")))]
    pub fn init(&mut self, executor_cnt: usize) {
        // WIP mute vulkan for now, tune with mock system
        let msg1: LoadfreeMessage<ActTensorTypes> =
            build_loadfree_msg!("spawn", "mock", executor_cnt);
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::LoadfreeMSG(msg1))
                .await;
        })
    }

    #[cfg(all(feature = "blas", not(feature = "mock"), not(feature = "vulkan")))]
    pub fn init(&mut self, executor_cnt: usize) {
        // WIP mute vulkan for now, tune with mock system
        let msg1: LoadfreeMessage<ActTensorTypes> =
            build_loadfree_msg!("spawn", "blas", executor_cnt);
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::LoadfreeMSG(msg1))
                .await;
        })
    }

    #[cfg(all(not(feature = "mock"), not(feature = "blas"), feature = "vulkan"))]
    pub fn init(&mut self, executor_cnt: usize) {
        // WIP mute vulkan for now, tune with mock system
        let msg1: LoadfreeMessage<ActTensorTypes> =
            build_loadfree_msg!("spawn", "vulkan", executor_cnt);
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::LoadfreeMSG(msg1))
                .await;
        })
    }

    #[cfg(all(feature = "mock", feature = "vulkan"))]
    pub fn init(&mut self, executor_cnt: usize) {
        // WIP mute vulkan for now, tune with mock system
        // TODO need fix, how to sort out the correct proposition between two type of backends
        let msg1: LoadfreeMessage<ActTensorTypes> =
            build_loadfree_msg!("spawn", "mock", executor_cnt);
        let msg2: LoadfreeMessage<ActTensorTypes> =
            build_loadfree_msg!("spawn", "vulkan", executor_cnt);
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::LoadfreeMSG(msg1))
                .await;
            self.actor_system
                .issue_order(RaptorMessage::LoadfreeMSG(msg2))
                .await;
        })
    }

    // TODO exec_mode = 
    // 0u8, eager + blocking + owned
    // 1u8, eager + blocking + borrowed
    // 2u8, eager + non-blocking + borrowed
    // 3u8, lazy
    pub fn launch_blocking_unary_compute(
        &mut self,
        opcode: CRTOpCode,
        in_tensor: ActTensorTypes,
    ) -> ActTensorTypes {
        let (send, recv) = oneshot::channel();
        let opmsg = PayloadMessage::UnaryComputeFunctorMsg {
            op: opcode,
            inp: in_tensor,
            respond_to: send,
        };
        info!(
            "::launch_blocking_unary_compute::send msg to actor_system {:?}",
            opmsg
        );
        debug!(
            "::launch_blocking_unary_compute::send msg to actor_system {:#?}",
            opmsg
        );
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::PayloadMSG(opmsg))
                .await;
        });

        let out_tensor = recv.blocking_recv().expect("no result after compute");
        info!("::blocking_recv done with result {:?}", out_tensor);
        debug!("::blocking_recv done with result {:#?}", out_tensor);
        out_tensor
    }

    pub fn launch_binary_compute(
        &mut self,
        opcode: CRTOpCode,
        lhs_tensor: ActTensorTypes,
        rhs_tensor: ActTensorTypes,
    ) -> ActTensorTypes {
        let (send, recv) = oneshot::channel();
        let opmsg = PayloadMessage::ComputeFunctorMsg {
            op: opcode,
            lhs: lhs_tensor,
            rhs: rhs_tensor,
            respond_to: send,
        };
        info!(
            "::launch_binary_compute::send msg to actor_system {:?}",
            opmsg
        );
        debug!(
            "::launch_binary_compute::send msg to actor_system {:#?}",
            opmsg
        );
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::PayloadMSG(opmsg))
                .await;
        });

        let out_tensor = recv.blocking_recv().expect("no result after compute");
        info!("::blocking_recv done with result {:?}", out_tensor);
        debug!("::blocking_recv done with result {:#?}", out_tensor);
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

    #[cfg(not(feature = "mock"))]
    #[test]
    fn test_e2e_add() {
        let mut se = HostSession::new();
        se.init(2);
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![11.0, 13.0, 17.0];
        let lhs_shape = vec![lhs.len()];
        let rhs_shape = vec![lhs.len()];
        // create lhs dataview
        let lhs_tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape),
        };
        let rhs_tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape),
        };
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
        let opcode = CRTOpCode::ADDF32;
        let mut result_buffer = se.launch_binary_compute(opcode, lhs_tensor_view, rhs_tensor_view);
        assert_eq!(
            result_buffer,
            ActTensorTypes::F32Tensor {
                data: TensorView::<f32>::new(vec!(12.0, 15.0, 20.0), ElementType::F32, vec![3])
                    .into(),
            }
        );
    }

    #[cfg(not(feature = "mock"))]
    #[test]
    fn test_e2e_sub() {
        let mut se = HostSession::new();
        se.init(2);
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![11.0, 13.0, 17.0];
        let lhs_shape = vec![lhs.len()];
        let rhs_shape = vec![lhs.len()];
        // create lhs dataview
        let lhs_tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape),
        };
        let rhs_tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape),
        };
        let opcode = CRTOpCode::SUBF32;
        let mut result_buffer = se.launch_binary_compute(opcode, lhs_tensor_view, rhs_tensor_view);
        assert_eq!(
            result_buffer,
            ActTensorTypes::F32Tensor {
                data: TensorView::<f32>::new(vec!(-10.0, -11.0, -14.0), ElementType::F32, vec![3])
                    .into(),
            }
        );
    }

    #[cfg(not(feature = "mock"))]
    #[test]
    fn test_e2e_matmul() {
        let mut se = HostSession::new();
        se.init(2);
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let lhs_shape = vec![2, 3];
        let rhs_shape = vec![3, 2];
        //create lhs dataview
        let lhs_tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape),
        };
        let rhs_tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape),
        };
        let opcode = CRTOpCode::MATMULF32;
        let mut result_buffer = se.launch_binary_compute(opcode, lhs_tensor_view, rhs_tensor_view);
        assert_eq!(
            result_buffer,
            ActTensorTypes::F32Tensor {
                data: TensorView::<f32>::new(
                    vec!(58.0, 64.0, 139.0, 154.0),
                    ElementType::F32,
                    vec![2, 2]
                )
                .into(),
            }
        );
    }
}
