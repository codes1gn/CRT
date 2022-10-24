use std::sync::{Arc, RwLock};
use std::{borrow::Cow, env, fs, iter, path::Path, ptr, slice, str::FromStr};

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
            .worker_threads(8)
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

    // TODO extract into utils
    fn build_notifiers_and_ready_checkers(
        &self,
        cnt: u8,
    ) -> (Vec<oneshot::Sender<u8>>, Vec<oneshot::Receiver<u8>>) {
        let mut senders = vec![];
        let mut receivers = vec![];
        for i in 0..cnt {
            let (send, recv) = oneshot::channel::<u8>();
            senders.push(send);
            receivers.push(recv);
        }
        (senders, receivers)
    }

    pub fn launch_dma_operation_inplace(
        &mut self,
        opcode: CRTOpCode,
        in_tensor: Arc<RwLock<ActTensorTypes>>,
        raw_shape_vec: Vec<usize>,
        signal_box: oneshot::Receiver<u8>,
        operand_out: usize,
    ) -> Vec<oneshot::Receiver<u8>> {
        // check readiness
        self.async_runtime.block_on(async {
            signal_box.await;
        });

        // set target shape
        match *in_tensor.write().unwrap() {
            ActTensorTypes::F32Tensor { ref mut data } => {
                let lhs_elements: usize = data.shape.iter().product();
                let rhs_elements: usize = raw_shape_vec.iter().product();
                assert_eq!(lhs_elements, rhs_elements);
                data.shape = raw_shape_vec;
            }
            _ => panic!("not support int types"),
        }

        // set notifier ready then put to context
        let (notifiers, recv_boxes) = self.build_notifiers_and_ready_checkers(4);
        for _notifier in notifiers.into_iter() {
            _notifier.send(0u8);
        }
        info!(
            "::vm::set tensor #{} OP^{:?} ready -- DATADEP",
            operand_out, opcode
        );
        recv_boxes
    }

    pub fn launch_dma_operation(
        &mut self,
        opcode: CRTOpCode,
        in_tensor: Arc<RwLock<ActTensorTypes>>,
        out_tensor: Arc<RwLock<ActTensorTypes>>,
        raw_shape_vec: Vec<usize>,
        signal_box: oneshot::Receiver<u8>,
        respond_id: usize,
        dev_at: Option<u8>,
    ) -> Vec<oneshot::Receiver<u8>> {
        // Action done here, and mock a dma operation done in async fashion
        // let from_data = match *in_tensor.read().unwrap() {
        //     ActTensorTypes::F32Tensor { ref data } => {
        //         let lhs_elements: usize = data.shape.iter().product();
        //         let rhs_elements: usize = raw_shape_vec.iter().product();
        //         assert_eq!(lhs_elements, rhs_elements);
        //         data.data.clone()
        //     }
        //     _ => panic!("not support int types"),
        // };
        // match *out_tensor.write().unwrap() {
        //     ActTensorTypes::F32Tensor { ref mut data } => {
        //         data.data = from_data;
        //         data.shape = raw_shape_vec.clone();
        //     }
        //     _ => panic!("not support int types"),
        // }

        // Mock the dma via Raptors system
        let (notifiers, recv_boxes) = self.build_notifiers_and_ready_checkers(4);
        let opmsg = PayloadMessage::DMAOperationMsg {
            op: opcode,
            inp: in_tensor,
            out: out_tensor,
            inp_ready_checker: signal_box,
            respond_to: notifiers,
            respond_id: respond_id,
            dev_at: dev_at,
            shape: raw_shape_vec,
        };
        debug!(
            "::launch_non_blocking_unary_compute::send msg to actor_system {:#?}",
            opmsg
        );
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::PayloadMSG(opmsg))
                .await;
        });
        info!(
            "::vm::set tensor #{} OP^{:?} ready -- DATADEP",
            respond_id, opcode
        );
        recv_boxes
    }

    // TODO exec_mode =
    // 0u8, eager + blocking + owned
    // 1u8, eager + blocking + borrowed
    // 2u8, eager + non-blocking + borrowed
    // 3u8, lazy
    pub fn launch_non_blocking_unary_compute(
        &mut self,
        opcode: CRTOpCode,
        in_tensor: Arc<RwLock<ActTensorTypes>>,
        out_tensor: Arc<RwLock<ActTensorTypes>>,
        signal_box: oneshot::Receiver<u8>,
        respond_id: usize,
        dev_at: Option<u8>,
    ) -> Vec<oneshot::Receiver<u8>> {
        // assume only one consumer after
        let (notifiers, ready_checkers) = self.build_notifiers_and_ready_checkers(4);
        let opmsg = PayloadMessage::NonRetUnaryComputeFunctorMsg {
            op: opcode,
            inp: in_tensor,
            out: out_tensor,
            inp_ready_checker: signal_box,
            respond_to: notifiers,
            respond_id: respond_id,
            dev_at: dev_at,
        };
        debug!(
            "::launch_non_blocking_unary_compute::send msg to actor_system {:#?}",
            opmsg
        );
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::PayloadMSG(opmsg))
                .await;
        });

        info!("::Non-blocking-launching Finished, return recv_box");
        return ready_checkers;
    }

    pub fn launch_non_blocking_binary_compute(
        &mut self,
        opcode: CRTOpCode,
        lhs_tensor: Arc<RwLock<ActTensorTypes>>,
        rhs_tensor: Arc<RwLock<ActTensorTypes>>,
        out_tensor: Arc<RwLock<ActTensorTypes>>,
        lhs_signal_box: oneshot::Receiver<u8>,
        rhs_signal_box: oneshot::Receiver<u8>,
        respond_id: usize,
        dev_at: Option<u8>,
    ) -> Vec<oneshot::Receiver<u8>> {
        // assume only one consumer after
        let (notifiers, ready_checkers) = self.build_notifiers_and_ready_checkers(4);
        let opmsg = PayloadMessage::NonRetBinaryComputeFunctorMsg {
            op: opcode,
            lhs: lhs_tensor,
            rhs: rhs_tensor,
            out: out_tensor,
            lhs_ready_checker: lhs_signal_box,
            rhs_ready_checker: rhs_signal_box,
            respond_to: notifiers,
            respond_id: respond_id,
            dev_at: dev_at,
        };
        debug!(
            "::launch_non_blocking_binary_compute::send msg to actor_system {:#?}",
            opmsg
        );
        self.async_runtime.block_on(async {
            self.actor_system
                .issue_order(RaptorMessage::PayloadMSG(opmsg))
                .await;
        });

        info!("::Non-blocking-launching Finished, return recv_box");
        return ready_checkers;
    }

    pub fn launch_blocking_unary_compute(
        &mut self,
        opcode: CRTOpCode,
        in_tensor: Arc<RwLock<ActTensorTypes>>,
    ) -> ActTensorTypes {
        let (send, recv) = oneshot::channel();
        let opmsg = PayloadMessage::UnaryComputeFunctorMsg {
            op: opcode,
            inp: in_tensor,
            respond_to: send,
        };
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

    pub fn launch_blocking_binary_compute(
        &mut self,
        opcode: CRTOpCode,
        lhs_tensor: Arc<RwLock<ActTensorTypes>>,
        rhs_tensor: Arc<RwLock<ActTensorTypes>>,
    ) -> ActTensorTypes {
        let (send, recv) = oneshot::channel();
        let opmsg = PayloadMessage::ComputeFunctorMsg {
            op: opcode,
            lhs: lhs_tensor,
            rhs: rhs_tensor,
            respond_to: send,
        };
        debug!(
            "::launch_blocking_binary_compute::send msg to actor_system {:#?}",
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
        let lhs_tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape),
        }));
        let rhs_tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape),
        }));
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
        let mut result_buffer = se.launch_blocking_binary_compute(
            opcode,
            Arc::clone(&lhs_tensor_view),
            Arc::clone(&rhs_tensor_view),
        );
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
        let lhs_tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape),
        }));
        let rhs_tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape),
        }));
        let opcode = CRTOpCode::SUBF32;
        let mut result_buffer = se.launch_blocking_binary_compute(
            opcode,
            Arc::clone(&lhs_tensor_view),
            Arc::clone(&rhs_tensor_view),
        );
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
        let lhs_tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(lhs, ElementType::F32, lhs_shape),
        }));
        let rhs_tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(rhs, ElementType::F32, rhs_shape),
        }));
        let opcode = CRTOpCode::MATMULF32;
        let mut result_buffer = se.launch_blocking_binary_compute(
            opcode,
            Arc::clone(&lhs_tensor_view),
            Arc::clone(&rhs_tensor_view),
        );
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
