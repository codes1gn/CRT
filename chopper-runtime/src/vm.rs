use multimap::MultiMap;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

#[cfg(any(feature = "blas", feature = "mock"))]
use rublas::prelude::*;
use tracing::{debug, info};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::{thread, time};

use crate::base::errors::*;
use crate::base::*;
use crate::buffer_types::*;
use crate::instance::*;
use crate::instruction::CRTOpCode;
use crate::session::*;
use crate::tensors::*;

// TODO may replace status with a enum
pub enum ExecMode {
    EAGER,
    LAZY,
}

#[derive(Debug)]
pub struct VM {
    registers: [i32; 32],
    inst_buffer: Vec<u8>,
    // use usize since this is decided by the arch of computer
    // 64/32 bits, it is equivalent to unsigned long long in C/C++
    // for a program counter, must use this to enumerate the reals.
    program_counter: usize,
    tensor_pool: HashMap<usize, Arc<RwLock<ActTensorTypes>>>,
    // TODO rename to subscriber
    subscribers: MultiMap<usize, oneshot::Receiver<u8>>,
    session: HostSession,
}

impl Drop for VM {
    fn drop(&mut self) {
        unsafe {
            info!("CRT-VM dropping");
        };
    }
}

impl VM {
    pub fn new() -> VM {
        let mut session = HostSession::new();
        VM {
            registers: [0; 32],
            program_counter: 0,
            inst_buffer: vec![],
            session: session,
            tensor_pool: HashMap::new(),
            subscribers: MultiMap::with_capacity(128),
        }
    }

    pub fn init(&mut self, executor_cnt: usize) {
        self.session.init(executor_cnt);
    }

    // getter
    pub fn inst_buffer(&self) -> &Vec<u8> {
        &self.inst_buffer
    }

    pub fn push_instruction(&mut self, byte: u8) {
        self.inst_buffer.push(byte);
    }

    // getter
    pub fn registers(&self) -> &[i32] {
        &self.registers
    }

    fn fetch_instruction(&mut self) -> Result<CRTOpCode, RuntimeStatusError> {
        if self.program_counter > self.inst_buffer.len() {
            return Err(RuntimeStatusError::EXEC_FINISH);
        }
        let opcode = CRTOpCode::from(self.decode_u8());
        info!("::vm::execute-instruction {:#?}", opcode);
        Ok(opcode)
    }

    fn decode_u8(&mut self) -> u8 {
        let _cmd_buffer = self.inst_buffer[self.program_counter];
        self.program_counter += 1;
        _cmd_buffer
    }

    fn decode_u16(&mut self) -> u16 {
        let encoded = vec![
            self.inst_buffer[self.program_counter],
            self.inst_buffer[self.program_counter + 1],
        ];
        self.program_counter += 2;
        let decoded: u16 = bincode::deserialize(&encoded).unwrap();
        decoded
    }

    fn decode_vec_len(&mut self) -> u16 {
        self.decode_u16()
    }

    fn decode_f32(&mut self) -> f32 {
        let encoded = self.decode_4_bytes();
        let decoded: f32 = bincode::deserialize(&encoded).unwrap();
        decoded
    }

    fn decode_f32_vec(&mut self, n_bytes: usize) -> Vec<f32> {
        let mut encoded: Vec<u8> = vec![];
        for _ in 0..n_bytes {
            encoded.push(self.inst_buffer[self.program_counter]);
            self.program_counter += 1;
        }
        let decoded: Vec<f32> = bincode::deserialize(&encoded).unwrap();
        decoded
    }

    fn decode_usize_vec(&mut self, n_bytes: usize) -> Vec<usize> {
        let mut encoded: Vec<u8> = vec![];
        for _ in 0..n_bytes {
            encoded.push(self.inst_buffer[self.program_counter]);
            self.program_counter += 1;
        }
        let decoded: Vec<usize> = bincode::deserialize(&encoded).unwrap();
        decoded
    }

    fn decode_4_bytes(&mut self) -> [u8; 4] {
        let mut ret_bytes = [0; 4];
        ret_bytes[0] = self.inst_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[1] = self.inst_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[2] = self.inst_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[3] = self.inst_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes
    }

    // TODO extract to util
    fn build_notifiers_and_subscribers(
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

    fn get_subscriber_at_pos(&mut self, pos: usize) -> oneshot::Receiver<u8> {
        self.subscribers
            .get_vec_mut(&pos)
            .unwrap()
            .remove(0 as usize)
    }

    // If contains subscriber already, replace it
    fn set_subscriber_at_pos(&mut self, pos: usize, subscribers: Vec<oneshot::Receiver<u8>>) {
        match self.subscribers.contains_key(&pos) {
            true => {
                self.subscribers.remove(&pos);
                self.subscribers.insert_many(pos, subscribers);
            }
            false => {
                self.subscribers.insert_many(pos, subscribers);
            }
        }
    }

    pub fn get_raw_vec_i32(&self, index: usize) -> Vec<i32> {
        // https://stackoverflow.com/questions/63501380/how-to-get-rid-of-cannot-return-value-referencing-temporary-value-error
        // https://stackoverflow.com/questions/32682876/is-there-any-way-to-return-a-reference-to-a-variable-created-in-a-function
        match *self.tensor_pool[&index].read().unwrap() {
            ActTensorTypes::I32Tensor { ref data } => data.data.clone(),
            _ => panic!("not support int types"),
        }
    }

    // TODO renaming
    pub fn get_raw_vec_f32(&self, index: usize) -> Vec<f32> {
        // TODO add guard for array-access-by-index
        match *self.tensor_pool[&index].read().unwrap() {
            ActTensorTypes::F32Tensor { ref data } => data.data.clone(),
            _ => panic!("not support int types"),
        }
    }

    // TODO to be moved into parametric arguments => push_data<T>(data: Vec<T>)
    pub fn push_tensor_i32(&mut self, index: usize, data: Vec<i32>) {
        let data_shape = vec![data.len()];
        // TODO-fix hide devices under device_context level
        let tensor_view = Arc::new(RwLock::new(ActTensorTypes::I32Tensor {
            data: TensorView::<i32>::new(data, ElementType::I32, data_shape),
        }));
        self.tensor_pool.insert(index, tensor_view);
    }

    pub fn get_tensor(&mut self, index: &usize) -> Arc<RwLock<ActTensorTypes>> {
        Arc::clone(&self.tensor_pool[index])
    }

    pub fn dump_tensor_f32(&self, index: usize) {
        if self.tensor_pool.contains_key(&index) == false {
            panic!("not-found tensor at slot #{:?}", index);
        }
        match *self.tensor_pool[&index].read().unwrap() {
            ActTensorTypes::F32Tensor { ref data } => {
                data.clone().dump(index);
            }
            _ => panic!("not support int types for #{:?}", index),
        }
    }

    pub fn get_tensor_shape(&self, index: usize) -> Vec<usize> {
        match *self.tensor_pool[&index].read().unwrap() {
            ActTensorTypes::F32Tensor { ref data } => data.shape.clone(),
            _ => panic!("not support int types"),
        }
    }

    pub fn assert_tensor_shape(&mut self, index: usize, shape: Vec<usize>) -> () {
        match *self.tensor_pool[&index].write().unwrap() {
            ActTensorTypes::F32Tensor { ref data } => {
                let lhs_elements: usize = data.shape.iter().product();
                let rhs_elements: usize = shape.iter().product();
                assert_eq!(lhs_elements, rhs_elements);
            }
            _ => panic!("not support int types"),
        }
    }

    pub fn push_shaped_tensor_at_pos(&mut self, index: usize, data: Vec<f32>, shape: Vec<usize>) {
        let tensor_view = Arc::new(RwLock::new(ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(data, ElementType::F32, shape),
        }));
        self.tensor_pool.insert(index, tensor_view);
    }

    fn define_shaped_placeholder_at_pos(&mut self, pos: usize, shape: Vec<usize>) {
        self.push_shaped_tensor_at_pos(pos, vec![0f32; shape.iter().product()], vec![]);
    }

    fn get_shaped_placeholder_at_pos_or(
        &mut self,
        pos: usize,
        shape: Vec<usize>,
    ) -> Arc<RwLock<ActTensorTypes>> {
        if self.tensor_pool.contains_key(&pos) == false {
            self.define_shaped_placeholder_at_pos(pos, shape);
        }
        self.get_tensor(&pos)
    }

    fn step_impl(&mut self, exec_mode: ExecMode) -> Result<u8, RuntimeStatusError> {
        let _inst = self.fetch_instruction().unwrap();
        match _inst {
            CRTOpCode::HALT => {
                info!("::vm::halt-vm");
                Ok(1)
            }
            CRTOpCode::ILLEGAL => {
                info!("::vm::halt-with-Illegal-Instruction");
                // panic!("HALT with ILLEGAL-INST {:?}", RuntimeStatusError::RT_ERROR);
                Err(RuntimeStatusError::RT_ERROR)
            }
            CRTOpCode::RETV => {
                info!("::vm::return from module");
                let operand_ret = self.decode_u8() as usize;
                // clear subscriber
                // currently solution, wait for this retv subscriber then returns
                let _subscriber = self
                    .subscribers
                    .get_vec_mut(&operand_ret)
                    .expect(&format!("failed to fetch subscriber {}", operand_ret).to_string())
                    .remove(0 as usize);
                _subscriber
                    .blocking_recv()
                    .expect("return value computing not ready");
                info!("::vm::ret-value compute done");
                // clear data_buffer before return
                // TODO maybe we need a strategy to decide what results retains and drop
                // considering function calls in module
                self.tensor_pool.retain(|&k, _| k == operand_ret);
                info!("::vm::ret-value retain and return");
                Ok(2)
            }
            // TODO rename to loadu16
            CRTOpCode::LOAD => {
                let register_id = self.decode_u8() as usize;
                let operand = self.decode_u16() as u16;
                // note the registers is defaultly i32s
                self.registers[register_id] = operand as i32;
                // TODO change return of error code as error enum
                // TODO change into verbose string
                Ok(0)
            }
            CRTOpCode::EXPF32 => {
                let operand_out = self.decode_u8() as usize;
                let operand_in = self.decode_u8() as usize;
                let in_dataview = self.get_tensor(&operand_in);
                // TODO rename dataview into ActTensorTypes
                let opcode = CRTOpCode::EXPF32;
                match exec_mode {
                    // should deprecate since it is not a safe mode that consumes
                    // try to learn from functional, consumes inputs is also a side-effect
                    ExecMode::EAGER => {
                        // non-consuming-inputs-style + blocking-style
                        info!("::vm::call-session-launch-unary-compute eager+borrowed+blocking");
                        let outs = self
                            .session
                            .launch_blocking_unary_compute(opcode, in_dataview);
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        self.tensor_pool
                            .insert(operand_out, Arc::new(RwLock::new(outs)));
                        Ok(0)
                    }
                    ExecMode::LAZY => {
                        // non-consuming-inputs-style + non-blocking-style
                        info!(
                            "::vm::poll subscriber for tensor #{} -- DATADEP",
                            operand_in
                        );
                        let _subscriber = self
                            .subscribers
                            .get_vec_mut(&operand_in)
                            .expect(
                                &format!("failed to fetch subscriber {} -- DATADEP", operand_in)
                                    .to_string(),
                            )
                            .remove(0 as usize);

                        // create a future-ready tensor
                        // TODO change data part into Option with a None init
                        // let output_placeholder = self.create_placeholder(&operand_in);
                        info!("::create placeholder tensor for ret-value-tensor");
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        let shape = self.get_tensor_shape(operand_in).to_vec();
                        self.push_shaped_tensor_at_pos(
                            operand_out,
                            vec![0f32; shape.iter().product()],
                            shape,
                        );
                        let out_dataview = self.get_tensor(&operand_out);

                        info!("::vm::call-session-launch-unary-compute eager+borrowed+blocking");
                        let recv_box = self.session.launch_non_blocking_unary_compute(
                            opcode,
                            in_dataview,
                            out_dataview,
                            _subscriber,
                            operand_out,
                        );
                        // solution to redefinition, replace subscriber with new
                        if self.subscribers.contains_key(&operand_out) {
                            // legacy path
                            // panic!("variable {}, redefined error", operand_out);

                            // new path
                            self.subscribers.remove(&operand_out);
                            self.subscribers.insert_many(operand_out, recv_box);
                        } else {
                            self.subscribers.insert_many(operand_out, recv_box);
                        }
                        // recv_box.into_iter().map(|x| {
                        //     // info!("dfjdslfj => {:?}, {:?}", operand_out, x);
                        //     self.subscribers.insert(operand_out, x);
                        // });
                        // for i in 0..3 {
                        //     let recvb = recv_box.into_iter().next().unwrap();
                        //     self.subscribers.insert(operand_out, recvb);
                        // }
                        info!(
                            "::vm::store subscriber for tensor #{} -- DATADEP",
                            operand_out
                        );

                        Ok(0)
                    }
                    _ => panic!("unknown exec-mode"),
                }
            }
            CRTOpCode::ADDF32
            | CRTOpCode::ADDI32
            | CRTOpCode::SUBF32
            | CRTOpCode::SUBI32
            | CRTOpCode::MULF32
            | CRTOpCode::MULI32
            | CRTOpCode::DIVF32
            | CRTOpCode::FLOORDIVI32
            | CRTOpCode::MATMULF32 => {
                let operand_out = self.decode_u8() as usize;
                let operand_lhs = self.decode_u8() as usize;
                let operand_rhs = self.decode_u8() as usize;
                let lhs_dataview = self.get_tensor(&operand_lhs);
                let rhs_dataview = self.get_tensor(&operand_rhs);
                let opcode = _inst;
                match exec_mode {
                    ExecMode::EAGER => {
                        // non-consuming-inputs-style + blocking-style
                        info!("::vm::call-session-launch-unary-compute eager+borrowed+blocking");
                        let outs = self.session.launch_blocking_binary_compute(
                            opcode,
                            lhs_dataview,
                            rhs_dataview,
                        );
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        self.tensor_pool
                            .insert(operand_out, Arc::new(RwLock::new(outs)));
                        Ok(0)
                    }
                    ExecMode::LAZY => {
                        // non-consuming-inputs-style + non-blocking-style
                        // TODO extract this logic to simplify
                        info!(
                            "::vm::poll subscriber for tensor #{} OP^{:?} -- DATADEP",
                            operand_lhs, opcode
                        );
                        let lhs_subscriber = self.get_subscriber_at_pos(operand_lhs);

                        info!(
                            "::vm::poll subscriber for tensor #{} OP^{:?} -- DATADEP",
                            operand_rhs, opcode
                        );
                        let rhs_subscriber = self.get_subscriber_at_pos(operand_rhs);

                        info!("::create placeholder tensor for ret-value-tensor");
                        info!(
                            "::vm::store-ret-value with index #{:?} OP^{:?}",
                            operand_out, opcode
                        );
                        // insert the output_placeholder
                        let shape = self.get_tensor_shape(operand_lhs).to_vec();
                        self.push_shaped_tensor_at_pos(
                            operand_out,
                            vec![0f32; shape.iter().product()],
                            shape,
                        );
                        let out_placeholder = self.get_tensor(&operand_out);

                        info!("::vm::call-session-launch-unary-compute eager+borrowed+blocking");
                        let _subscribers = self.session.launch_non_blocking_binary_compute(
                            opcode,
                            lhs_dataview,
                            rhs_dataview,
                            out_placeholder,
                            lhs_subscriber,
                            rhs_subscriber,
                            operand_out,
                        );
                        if self.subscribers.contains_key(&operand_out) {
                            self.subscribers.remove(&operand_out);
                            self.subscribers.insert_many(operand_out, _subscribers);
                        } else {
                            self.subscribers.insert_many(operand_out, _subscribers);
                        }
                        info!(
                            "::vm::store subscriber for tensor #{} -- DATADEP",
                            operand_out
                        );

                        Ok(0)
                    }
                    _ => panic!("unknown exec-mode"),
                }
            }
            CRTOpCode::CONSTI32 => {
                // TODO do some action, add data_buffer
                // create lhs dataview
                // TODO enable it
                let operand_out = self.decode_u8() as usize;
                let operand_in = self.decode_4_bytes();
                let operand_in_i32 = i32::from_le_bytes(operand_in);
                self.push_tensor_i32(operand_out, vec![operand_in_i32]);
                Ok(0)
            }
            CRTOpCode::CONSTF32 => {
                // TODO do some action, add data_buffer
                // create lhs dataview
                let operand_out = self.decode_u8() as usize;
                let operand_in = self.decode_4_bytes();
                let operand_in_f32 = f32::from_le_bytes(operand_in);
                self.push_shaped_tensor_at_pos(operand_out, vec![operand_in_f32], vec![1]);
                Ok(0)
            }
            CRTOpCode::CONSTTENSOR => {
                let operand_out = self.decode_u8() as usize;
                let data_size = self.decode_vec_len() as usize;
                let raw_data_vec = self.decode_f32_vec(data_size);
                let shape_size = self.decode_vec_len() as usize;
                let raw_shape_vec = self.decode_usize_vec(shape_size);
                self.push_shaped_tensor_at_pos(operand_out, raw_data_vec, raw_shape_vec);
                Ok(0)
            }
            CRTOpCode::SVALUETENSOR => {
                let operand_out = self.decode_u8() as usize;
                let data_generator = self.decode_4_bytes();
                // TODO currently svalue is hardcoded as float
                let data_generator_f32 = f32::from_le_bytes(data_generator);
                let shape_size = self.decode_vec_len() as usize;
                let raw_shape_vec = self.decode_usize_vec(shape_size);
                info!(
                    "::vm::generate+store tensor-value with index #{:?}",
                    operand_out
                );
                self.push_shaped_tensor_at_pos(
                    operand_out,
                    vec![data_generator_f32; raw_shape_vec.iter().product()],
                    raw_shape_vec,
                );
                for i in 0..8 {
                    let (notifier, subscriber) = oneshot::channel::<u8>();
                    notifier.send(0u8);
                    self.subscribers.insert(operand_out, subscriber);
                }
                info!("::vm::fill data-subscriber #{}", operand_out);
                Ok(0)
            }
            CRTOpCode::RNGTENSOR => {
                let operand_out = self.decode_u8() as usize;
                let distribution = self.decode_u8();
                let shape_size = self.decode_vec_len() as usize;
                let raw_shape_vec = self.decode_usize_vec(shape_size);
                let _tensor = match distribution {
                    // TODO make min-max adjustable
                    0 => BlasTensor::uniform(raw_shape_vec.clone(), -1f32, 1f32),
                    1 => BlasTensor::normal(raw_shape_vec.clone(), 0f32, 1f32),
                    _ => panic!("unknown rng category????"),
                };
                self.push_shaped_tensor_at_pos(
                    operand_out,
                    TensorView::<f32>::from(_tensor).data,
                    raw_shape_vec,
                );
                Ok(0)
            }
            CRTOpCode::RESHAPE => {
                let operand_out = self.decode_u8() as usize;
                let operand_in = self.decode_u8() as usize;
                let shape_size = self.decode_vec_len() as usize;
                let raw_shape_vec = self.decode_usize_vec(shape_size);
                let in_tensor = self.get_tensor(&operand_in);
                let opcode = _inst;
                // if operand_in != operand_out
                match exec_mode {
                    // should deprecate since it is not a safe mode that consumes
                    // try to learn from functional, consumes inputs is also a side-effect
                    ExecMode::EAGER => match (operand_in == operand_out) {
                        true => {
                            match *in_tensor.write().unwrap() {
                                ActTensorTypes::F32Tensor { ref mut data } => {
                                    let lhs_elements: usize = data.shape.iter().product();
                                    let rhs_elements: usize = raw_shape_vec.iter().product();
                                    assert_eq!(lhs_elements, rhs_elements);
                                    data.shape = raw_shape_vec;
                                }
                                _ => panic!("not support int types"),
                            }
                            return Ok(0);
                        }
                        false => {
                            match *in_tensor.read().unwrap() {
                                ActTensorTypes::F32Tensor { ref data } => {
                                    let lhs_elements: usize = data.shape.iter().product();
                                    let rhs_elements: usize = raw_shape_vec.iter().product();
                                    assert_eq!(lhs_elements, rhs_elements);
                                    self.push_shaped_tensor_at_pos(
                                        operand_out,
                                        data.data.clone(),
                                        raw_shape_vec,
                                    );
                                }
                                _ => panic!("not support int types"),
                            }
                            return Ok(0);
                        }
                    },
                    ExecMode::LAZY => {
                        // non-consuming-inputs-style + non-blocking-style
                        info!(
                            "::vm::poll subscriber for tensor #{} OP^{:?} -- DATADEP",
                            operand_in, opcode
                        );
                        let _subscriber = self.get_subscriber_at_pos(operand_in);

                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        let recv_boxes = if operand_in == operand_out {
                            let recv_boxes = self.session.launch_dma_operation_inplace(
                                opcode,
                                in_tensor,
                                raw_shape_vec,
                                _subscriber,
                                operand_out,
                            );
                            recv_boxes
                        } else {
                            let out_placeholder = self.get_shaped_placeholder_at_pos_or(
                                operand_out,
                                raw_shape_vec.clone(),
                            );

                            let recv_boxes = self.session.launch_dma_operation(
                                opcode,
                                in_tensor,
                                out_placeholder,
                                raw_shape_vec,
                                _subscriber,
                                operand_out,
                            );
                            recv_boxes
                        };

                        self.set_subscriber_at_pos(operand_out, recv_boxes);
                        info!(
                            "::vm::store subscriber for tensor #{} OP^{:?} -- DATADEP",
                            operand_out, opcode
                        );

                        Ok(0)
                    }
                    _ => panic!("unknown exec-mode"),
                }
            }
            // MOCK tranpose is mocked with reshape
            #[cfg(feature = "mock")]
            CRTOpCode::TRANSPOSE => {
                let operand_out = self.decode_u8() as usize;
                let operand_in = self.decode_u8() as usize;
                let shape_size = self.decode_vec_len() as usize;
                let raw_shape_vec = self.decode_usize_vec(shape_size);
                let in_dataview = self.get_tensor(&operand_in);
                let opcode = _inst;
                // if operand_in != operand_out
                match exec_mode {
                    // should deprecate since it is not a safe mode that consumes
                    // try to learn from functional, consumes inputs is also a side-effect
                    ExecMode::EAGER => {
                        if operand_in == operand_out {
                            match *in_dataview.write().unwrap() {
                                ActTensorTypes::F32Tensor { ref mut data } => {
                                    let lhs_elements: usize = data.shape.iter().product();
                                    let rhs_elements: usize = raw_shape_vec.iter().product();
                                    assert_eq!(lhs_elements, rhs_elements);
                                    data.shape = raw_shape_vec;
                                }
                                _ => panic!("not support int types"),
                            }
                            return Ok(0);
                        } else {
                            match *in_dataview.read().unwrap() {
                                ActTensorTypes::F32Tensor { ref data } => {
                                    let lhs_elements: usize = data.shape.iter().product();
                                    let rhs_elements: usize = raw_shape_vec.iter().product();
                                    assert_eq!(lhs_elements, rhs_elements);
                                    self.push_shaped_tensor_at_pos(
                                        operand_out,
                                        data.data.clone(),
                                        raw_shape_vec,
                                    );
                                }
                                _ => panic!("not support int types"),
                            }
                            return Ok(0);
                        }
                    }
                    ExecMode::LAZY => {
                        // non-consuming-inputs-style + non-blocking-style
                        info!(
                            "::vm::poll subscriber for tensor #{} in OP^{:?} -- DATADEP",
                            operand_in, opcode
                        );
                        let _subscriber = self
                            .subscribers
                            .get_vec_mut(&operand_in)
                            .expect(
                                &format!(
                                    "failed to fetch subscriber {} OP^{:?} -- DATADEP",
                                    operand_in, opcode
                                )
                                .to_string(),
                            )
                            .remove(0 as usize);

                        // create a future-ready tensor
                        // TODO change data part into Option with a None init
                        // let output_placeholder = self.create_placeholder(&operand_in);
                        info!("::create placeholder tensor for ret-value-tensor");
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        if operand_in == operand_out {
                            match *in_dataview.write().unwrap() {
                                ActTensorTypes::F32Tensor { ref mut data } => {
                                    let lhs_elements: usize = data.shape.iter().product();
                                    let rhs_elements: usize = raw_shape_vec.iter().product();
                                    assert_eq!(lhs_elements, rhs_elements);
                                    data.shape = raw_shape_vec;
                                }
                                _ => panic!("not support int types"),
                            }
                        } else {
                            match *in_dataview.read().unwrap() {
                                ActTensorTypes::F32Tensor { ref data } => {
                                    let lhs_elements: usize = data.shape.iter().product();
                                    let rhs_elements: usize = raw_shape_vec.iter().product();
                                    assert_eq!(lhs_elements, rhs_elements);
                                    self.push_shaped_tensor_at_pos(
                                        operand_out,
                                        data.data.clone(),
                                        raw_shape_vec,
                                    );
                                }
                                _ => panic!("not support int types"),
                            }
                        }
                        // solution to redefinition, replace subscriber with new
                        let (notifiers, recv_boxes) = self.build_notifiers_and_subscribers(4);
                        if self.subscribers.contains_key(&operand_out) {
                            // legacy path
                            // panic!("variable {}, redefined error", operand_out);

                            // new path
                            self.subscribers.remove(&operand_out);
                            self.subscribers.insert_many(operand_out, recv_boxes);
                        } else {
                            self.subscribers.insert_many(operand_out, recv_boxes);
                        }
                        info!(
                            "::vm::store subscriber placeholder for tensor #{} OP^{:?} -- DATADEP",
                            operand_out, opcode
                        );

                        for _notifier in notifiers.into_iter() {
                            _notifier.send(0u8);
                        }
                        info!(
                            "::vm::set tensor #{} OP^{:?} ready -- DATADEP",
                            operand_out, opcode
                        );

                        Ok(0)
                    }
                    _ => panic!("unknown exec-mode"),
                }
            }
            _ => {
                panic!("Not Implemented Error Execution Step Code");
            }
        }
    }

    fn eager_step(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::eager-step");
        if self.program_counter >= self.inst_buffer.len() {
            info!("::vm::cmd-buffer empty >> halt");
            return Err(RuntimeStatusError::EXEC_FINISH);
        }
        self.step_impl(ExecMode::EAGER)
    }

    fn lazy_step(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::lazy-step");
        if self.program_counter >= self.inst_buffer.len() {
            info!("::vm::cmd-buffer empty >> halt");
            return Err(RuntimeStatusError::EXEC_FINISH);
        }
        self.step_impl(ExecMode::LAZY)
    }

    pub fn run_eagerly(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::run-eagerly");
        loop {
            let status = self.eager_step();
            match status {
                Ok(_) => {
                    info!("continue ");
                }
                Err(RuntimeStatusError::EXEC_FINISH) => return Ok(0),
                // add panic handle for illegal instruction
                Err(RuntimeStatusError::RT_ERROR) => panic!("Runtime Error Detected"),
                Err(_) => return status,
            }
        }
        info!("::vm::task-dispatch finished, sleep to wait all done");
    }

    pub fn run_lazily(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::run-lazily");
        loop {
            let status = self.lazy_step();
            match status {
                Ok(_) => {
                    info!("continue ");
                }
                Err(RuntimeStatusError::EXEC_FINISH) => return Ok(0),
                // add panic handle for illegal instruction
                Err(RuntimeStatusError::RT_ERROR) => panic!("Runtime Error Detected"),
                Err(_) => return status,
            }
        }
        info!("::vm::task-dispatch finished, sleep to wait all done");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_vm_struct() {
        let vm = VM::new();
        assert_eq!(vm.registers[0], 0);
    }

    // TODO maybe need to test the middle status when halt invoked in run-until-end way.
    #[test]
    fn test_halt_step() {
        let mut vm = VM::new();
        vm.init(2);
        vm.inst_buffer = vec![0, 0, 0];
        let exit_code = vm.eager_step();
        assert_eq!(exit_code.is_ok(), true);
        let u8_exit_code = exit_code.unwrap();
        assert_eq!(u8_exit_code, 1);
        assert_eq!(vm.program_counter, 1);
    }

    #[test]
    fn test_vm_dummy() {
        let mut vm = VM::new();
        vm.init(2);
        vm.inst_buffer = vec![];
        let exit_code = vm.eager_step();
        // TODO to use Ok(program_finish)
        assert_eq!(exit_code.is_err(), true);
        assert_eq!(vm.program_counter, 0);
    }

    #[test]
    fn test_vm_illegal() {
        let mut vm = VM::new();
        vm.inst_buffer = vec![255];
        vm.init(2);
        let exit_code = vm.eager_step();
        assert_eq!(exit_code.is_ok(), false);
        assert_eq!(vm.program_counter, 1);
    }

    #[test]
    fn test_vm_fetch_instruction() {
        let mut vm = VM::new();
        vm.inst_buffer = vec![0];
        let opcode = vm.fetch_instruction();
        assert_eq!(opcode.unwrap(), CRTOpCode::HALT);
    }

    #[test]
    fn test_vm_next_byte() {
        let mut vm = VM::new();
        vm.inst_buffer = vec![8];
        let data = vm.decode_u8();
        assert_eq!(data, 8);
    }
}
