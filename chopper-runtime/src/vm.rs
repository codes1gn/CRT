use multimap::MultiMap;
use std::collections::HashMap;
use std::{thread, time};

#[cfg(any(feature = "blas", feature = "mock"))]
use rublas::prelude::*;
use tracing::{debug, info};

use tokio::sync::oneshot;

use serde::{Deserialize, Serialize};

use crate::base::errors::*;
use crate::base::*;
use crate::instruction::CRTOpCode;

use crate::buffer_types::*;
use crate::instance::*;
use crate::session::*;
use crate::tensors::*;

#[derive(Debug)]
pub struct VM {
    registers: [i32; 32],
    command_buffer: Vec<u8>,
    // use usize since this is decided by the arch of computer
    // 64/32 bits, it is equivalent to unsigned long long in C/C++
    // for a program counter, must use this to enumerate the reals.
    program_counter: usize,
    // TODO to bring device instance into interpreter, may need to impl Default
    // to allow new without explicit value of Session, thus not borrow a moved
    // value -> device instance
    // pub data_buffer_f32: HashMap<usize, UniBuffer<concrete_backend::Backend, f32>>,
    pub data_buffer_f32: HashMap<usize, ActTensorTypes>,
    // pub data_buffer_i32: HashMap<usize, UniBuffer<concrete_backend::Backend, i32>>,
    // TODO maybe remove
    pub data_buffer_i32: HashMap<usize, ActTensorTypes>,
    pub ready_checkers: MultiMap<usize, oneshot::Receiver<u8>>,
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
            command_buffer: vec![],
            session: session,
            data_buffer_f32: HashMap::new(),
            data_buffer_i32: HashMap::new(),
            ready_checkers: MultiMap::with_capacity(128),
        }
    }

    pub fn init(&mut self, executor_cnt: usize) {
        self.session.init(executor_cnt);
    }

    fn fetch_instruction(&mut self) -> Result<CRTOpCode, RuntimeStatusError> {
        if self.program_counter > self.command_buffer.len() {
            return Err(RuntimeStatusError::EXEC_FINISH);
        }
        let opcode = CRTOpCode::from(self.command_buffer[self.program_counter]);
        info!("::vm::execute-instruction {:#?}", opcode);
        self.program_counter += 1;
        Ok(opcode)
    }

    // TODO pack below functions into decode a tensor
    fn decode_two_bytes_as_vec_size(&mut self) -> u16 {
        let encoded = vec![
            self.command_buffer[self.program_counter],
            self.command_buffer[self.program_counter + 1],
        ];
        // println!("decoding data len {:?}", encoded);
        self.program_counter += 2;
        let decoded: u16 = bincode::deserialize(&encoded).unwrap();
        decoded
    }

    fn decode_n_bytes_as_f32_vec(&mut self, lens: usize) -> Vec<f32> {
        let mut encoded: Vec<u8> = vec![];
        for _ in 0..lens {
            encoded.push(self.command_buffer[self.program_counter]);
            self.program_counter += 1;
        }
        // println!("decoding data bytes {:?}", encoded);
        let decoded: Vec<f32> = bincode::deserialize(&encoded).unwrap();
        // println!("{:?}", decoded);
        decoded
    }

    fn decode_n_bytes_as_usize_vec(&mut self, lens: usize) -> Vec<usize> {
        let mut encoded: Vec<u8> = vec![];
        for _ in 0..lens {
            encoded.push(self.command_buffer[self.program_counter]);
            self.program_counter += 1;
        }
        // println!("decoding shape bytes {:?}", encoded);
        let decoded: Vec<usize> = bincode::deserialize(&encoded).unwrap();
        // println!("{:?}", decoded);
        decoded
    }

    // refactor into get_<type> form, make it more verbose
    fn get_next_byte(&mut self) -> u8 {
        let _cmd_buffer = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        _cmd_buffer
    }

    fn get_next_two_bytes(&mut self) -> u16 {
        let _cmd_buffer = ((self.command_buffer[self.program_counter] as u16) << 8)
            | self.command_buffer[self.program_counter + 1] as u16;
        self.program_counter += 2;
        _cmd_buffer
    }

    fn get_next_four_bytes(&mut self) -> [u8; 4] {
        let mut ret_bytes = [0; 4];
        ret_bytes[0] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[1] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[2] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[3] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes
    }

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

    // TODO may replace status with a enum
    // TODO may replace exec_mode with enum
    // TODO exec_mode =
    // 0u8, eager + blocking + consuming-inputs
    // 1u8, eager + blocking + non-consuming-inputs
    // 2u8, eager + non-blocking + non-consuming-inputs
    // 3u8, lazy
    fn step_impl(&mut self, exec_mode: u8) -> Result<u8, RuntimeStatusError> {
        info!("::vm::execute step eagerly");
        match self.fetch_instruction().unwrap() {
            CRTOpCode::HALT => {
                info!("::vm::halt-vm");
                Ok(1)
            }
            CRTOpCode::ILLEGAL => {
                info!("::vm::halt-with-Illegal-Instruction");
                Err(RuntimeStatusError::RT_ERROR)
            }
            CRTOpCode::RETV => {
                info!("::vm::return from module");
                let operand_ret = self.get_next_byte() as usize;
                // clear ready_checker
                // currently solution, wait for this retv ready-checker then returns
                println!("{}", operand_ret);
                let ready_checker = self
                    .ready_checkers
                    .get_vec_mut(&operand_ret)
                    .expect(&format!("failed to fetch ready-checker {}", operand_ret).to_string())
                    .remove(0 as usize);
                ready_checker
                    .blocking_recv()
                    .expect("return value computing not ready");
                info!("::vm::ret-value compute done");
                // clear data_buffer before return
                // TODO maybe we need a strategy to decide what results retains and drop
                // considering function calls in module
                self.data_buffer_f32.retain(|&k, _| k == operand_ret);
                info!("::vm::ret-value retain and return");
                Ok(2)
            }
            // TODO rename to loadu16
            CRTOpCode::LOAD => {
                let register_id = self.get_next_byte() as usize;
                let operand = self.get_next_two_bytes() as u16;
                // note the registers is defaultly i32s
                self.registers[register_id] = operand as i32;
                // TODO change return of error code as error enum
                // TODO change into verbose string
                Ok(0)
            }
            CRTOpCode::EXPF32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_in = self.get_next_byte() as usize;
                // TODO rename dataview into ActTensorTypes
                let opcode = CRTOpCode::EXPF32;
                match exec_mode {
                    // should deprecate since it is not a safe mode that consumes
                    // try to learn from functional, consumes inputs is also a side-effect
                    0u8 => {
                        // consuming-inputs-style + blocking-style
                        let in_dataview = self.data_buffer_f32.remove(&operand_in).unwrap();
                        info!("::vm::call-session-launch-unary-compute eager+owned+blocking");
                        let outs = self
                            .session
                            .launch_blocking_unary_compute(opcode, in_dataview);
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        self.data_buffer_f32.insert(operand_out, outs);
                        Ok(0)
                    }
                    1u8 => {
                        // non-consuming-inputs-style + blocking-style
                        let in_dataview_clone = self.get_data_clone(&operand_in);
                        info!("::vm::call-session-launch-unary-compute eager+borrowed+blocking");
                        let outs = self
                            .session
                            .launch_blocking_unary_compute(opcode, in_dataview_clone);
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        self.data_buffer_f32.insert(operand_out, outs);
                        Ok(0)
                    }
                    2u8 => {
                        // non-consuming-inputs-style + non-blocking-style
                        println!("{:#?}", self.ready_checkers);
                        let in_dataview_clone = self.get_data_clone(&operand_in);
                        info!("::vm::poll ready-checker for tensor #{}", operand_in);
                        let ready_checker = self
                            .ready_checkers
                            .get_vec_mut(&operand_in)
                            .expect(
                                &format!("failed to fetch ready-checker {}", operand_in)
                                    .to_string(),
                            )
                            .remove(0 as usize);
                        info!("::vm::call-session-launch-unary-compute eager+borrowed+blocking");
                        let recv_box = self.session.launch_non_blocking_unary_compute(
                            opcode,
                            in_dataview_clone,
                            ready_checker,
                            operand_out,
                        );
                        // TODO check redefinition of operand-out
                        if self.ready_checkers.contains_key(&operand_out) {
                            panic!("variable {}, redefined error", operand_out);
                        }
                        self.ready_checkers.insert_many(operand_out, recv_box);
                        // recv_box.into_iter().map(|x| {
                        //     // info!("dfjdslfj => {:?}, {:?}", operand_out, x);
                        //     self.ready_checkers.insert(operand_out, x);
                        // });
                        // for i in 0..3 {
                        //     let recvb = recv_box.into_iter().next().unwrap();
                        //     self.ready_checkers.insert(operand_out, recvb);
                        // }
                        info!("::vm::store ready-checker for tensor #{}", operand_out);

                        // create a future-ready tensor
                        // TODO change data part into Option with a None init
                        let output_placeholder_clone = self.get_data_clone(&operand_in);
                        info!("::create placeholder tensor for ret-value-tensor");
                        info!("::vm::store-ret-value with index #{:?}", operand_out);
                        self.data_buffer_f32
                            .insert(operand_out, output_placeholder_clone);
                        println!("{:#?}", self.ready_checkers);
                        Ok(0)
                    }
                    _ => {
                        panic!("lsls")
                    }
                }
            }
            CRTOpCode::ADDI32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::ADDI32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                // .benchmark_run::<i32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::ADDF32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::ADDF32;
                // TODO launch_binary_compute is a blocking method that recv's compute ret-value
                // in blocking fashion
                // TODO impl a non-blocking version
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                // .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::SUBI32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::SUBI32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::SUBF32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::SUBF32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                // .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::MULI32 => {
                // TODO merge logics together and pass the CRTOpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::MULI32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::MULF32 => {
                // TODO merge logics together and pass the CRTOpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::MULF32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                // .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::FLOORDIVI32 => {
                // TODO merge logics together and pass the CRTOpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::FLOORDIVI32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::DIVF32 => {
                // TODO merge logics together and pass the CRTOpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::DIVF32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                // .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::MATMULF32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = CRTOpCode::MATMULF32;
                let outs = self
                    .session
                    .launch_binary_compute(opcode, lhs_dataview, rhs_dataview);
                // .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            CRTOpCode::CONSTI32 => {
                // TODO do some action, add data_buffer
                // create lhs dataview
                // TODO enable it
                let operand_out = self.get_next_byte() as usize;
                let operand_in = self.get_next_four_bytes();
                let operand_in_i32 = i32::from_le_bytes(operand_in);
                self.push_data_buffer_i32(operand_out, vec![operand_in_i32]);
                Ok(0)
            }
            CRTOpCode::CONSTF32 => {
                // TODO do some action, add data_buffer
                // create lhs dataview
                let operand_out = self.get_next_byte() as usize;
                let operand_in = self.get_next_four_bytes();
                let operand_in_f32 = f32::from_le_bytes(operand_in);
                self.push_data_buffer_f32(operand_out, vec![operand_in_f32]);
                Ok(0)
            }
            CRTOpCode::CONSTTENSOR => {
                let operand_out = self.get_next_byte() as usize;
                let data_size = self.decode_two_bytes_as_vec_size() as usize;
                let raw_data_vec = self.decode_n_bytes_as_f32_vec(data_size);
                let shape_size = self.decode_two_bytes_as_vec_size() as usize;
                let raw_shape_vec = self.decode_n_bytes_as_usize_vec(shape_size);
                self.push_tensor_buffer(operand_out, raw_data_vec, raw_shape_vec);
                Ok(0)
            }
            CRTOpCode::SVALUETENSOR => {
                let operand_out = self.get_next_byte() as usize;
                let data_generator = self.get_next_four_bytes();
                // TODO currently svalue is hardcoded as float
                let data_generator_f32 = f32::from_le_bytes(data_generator);
                let shape_size = self.decode_two_bytes_as_vec_size() as usize;
                let raw_shape_vec = self.decode_n_bytes_as_usize_vec(shape_size);
                info!(
                    "::vm::generate+store tensor-value with index #{:?}",
                    operand_out
                );
                self.push_tensor_buffer(
                    operand_out,
                    vec![data_generator_f32; raw_shape_vec.iter().product()],
                    raw_shape_vec,
                );
                for i in 0..8 {
                    let (notifier, ready_checker) = oneshot::channel::<u8>();
                    notifier.send(0u8);
                    self.ready_checkers.insert(operand_out, ready_checker);
                }
                info!("::vm::fill data-ready-checker #{}", operand_out);
                Ok(0)
            }
            CRTOpCode::RNGTENSOR => {
                let operand_out = self.get_next_byte() as usize;
                let distribution = self.get_next_byte();
                let shape_size = self.decode_two_bytes_as_vec_size() as usize;
                let raw_shape_vec = self.decode_n_bytes_as_usize_vec(shape_size);
                let _tensor = match distribution {
                    // TODO make min-max adjustable
                    0 => BlasTensor::uniform(raw_shape_vec.clone(), -1f32, 1f32),
                    1 => BlasTensor::normal(raw_shape_vec.clone(), 0f32, 1f32),
                    _ => panic!("unknown rng category????"),
                };
                self.push_tensor_buffer(
                    operand_out,
                    TensorView::<f32>::from(_tensor).data,
                    raw_shape_vec,
                );
                Ok(0)
            }
            _ => {
                panic!("Not Implemented Error Execution Step Code");
            }
        }
    }

    // property functions that is public
    pub fn command_buffer(&self) -> &Vec<u8> {
        &self.command_buffer
    }

    // property functions that is public
    pub fn registers(&self) -> &[i32] {
        &self.registers
    }

    pub fn push_bytecode_into_cmdbuffer(&mut self, byte: u8) {
        self.command_buffer.push(byte);
    }

    pub fn get_idata(&self, index: usize) -> &Vec<i32> {
        match &self.data_buffer_i32[&index] {
            ActTensorTypes::I32Tensor { data } => &data.data,
            _ => panic!("not support int types"),
        }
    }

    // TODO to be moved into parametric arguments => push_data<T>(data: Vec<T>)
    pub fn push_data_buffer_i32(&mut self, index: usize, data: Vec<i32>) {
        let data_shape = vec![data.len()];
        // TODO-fix hide devices under device_context level
        let tensor_view = ActTensorTypes::I32Tensor {
            data: TensorView::<i32>::new(data, ElementType::I32, data_shape),
        };
        // TODO-trial lowering UniBuffer range, to make session dev independent
        // let mut data_buffer = UniBuffer::<concrete_backend::Backend, i32>::new(
        //     &self.session.device_context.device,
        //     &self
        //         .session
        //         .device_context
        //         .device_instance
        //         .memory_property()
        //         .memory_types,
        //     tensor_view,
        // );
        self.data_buffer_i32.insert(index, tensor_view);
    }

    pub fn get_data_clone(&mut self, index: &usize) -> ActTensorTypes {
        self.data_buffer_f32[index].clone()
    }

    // TODO renaming
    pub fn get_fdata(&self, index: usize) -> &Vec<f32> {
        // TODO add guard for array-access-by-index
        match &self.data_buffer_f32[&index] {
            ActTensorTypes::F32Tensor { data } => &data.data,
            _ => panic!("not support int types"),
        }
    }

    pub fn get_fshape(&self, index: usize) -> &Vec<usize> {
        match &self.data_buffer_f32[&index] {
            ActTensorTypes::F32Tensor { data } => &data.shape,
            _ => panic!("not support int types"),
        }
    }

    pub fn push_data_buffer_f32(&mut self, index: usize, data: Vec<f32>) {
        let data_shape = vec![data.len()];
        let tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(data, ElementType::F32, data_shape),
        };
        // TODO-trial lowering UniBuffer range, to make session dev independent
        // let mut data_buffer = UniBuffer::<concrete_backend::Backend, f32>::new(
        //     &self.session.device_context.device,
        //     &self
        //         .session
        //         .device_context
        //         .device_instance
        //         .memory_property()
        //         .memory_types,
        //     tensor_view,
        // );
        self.data_buffer_f32.insert(index, tensor_view);
    }

    pub fn push_tensor_buffer(&mut self, index: usize, data: Vec<f32>, shape: Vec<usize>) {
        let tensor_view = ActTensorTypes::F32Tensor {
            data: TensorView::<f32>::new(data, ElementType::F32, shape),
        };
        // TODO-trial lowering UniBuffer range, to make session dev independent
        // let mut data_buffer = UniBuffer::<concrete_backend::Backend, f32>::new(
        //     &self.session.device_context.device,
        //     &self
        //         .session
        //         .device_context
        //         .device_instance
        //         .memory_property()
        //         .memory_types,
        //     tensor_view,
        // );
        self.data_buffer_f32.insert(index, tensor_view);
    }

    // entry functions for execute, that is public
    pub fn eager_step(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::eager-step");
        if self.program_counter >= self.command_buffer.len() {
            info!("::vm::cmd-buffer empty >> halt");
            return Err(RuntimeStatusError::EXEC_FINISH);
        }
        self.step_impl(1 as u8)
    }

    // entry functions for execute, that is public
    pub fn lazy_step(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::eager-step");
        if self.program_counter >= self.command_buffer.len() {
            info!("::vm::cmd-buffer empty >> halt");
            return Err(RuntimeStatusError::EXEC_FINISH);
        }
        // TODO exec_mode =
        // 0u8, eager + blocking + owned
        // 1u8, eager + blocking + borrowed
        // 2u8, eager + non-blocking + borrowed
        // 3u8, lazy
        // self.step_impl(0 as u8)
        // self.step_impl(1 as u8)
        self.step_impl(2 as u8)
    }

    // TODO modify the return into statuscode
    pub fn run_eagerly(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::run-eagerly");
        loop {
            let status = self.eager_step();
            match status {
                Ok(_) => {
                    info!("continue ");
                }
                Err(_) => return status,
            }
        }
        info!("::vm::task-dispatch finished, sleep to wait all done");
    }

    // TODO modify the return into statuscode
    pub fn run_lazily(&mut self) -> Result<u8, RuntimeStatusError> {
        info!("::vm::run-lazily");
        loop {
            let status = self.lazy_step();
            match status {
                Ok(_) => {
                    info!("continue ");
                }
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
        vm.command_buffer = vec![0, 0, 0];
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
        vm.command_buffer = vec![];
        let exit_code = vm.eager_step();
        // TODO to use Ok(program_finish)
        assert_eq!(exit_code.is_err(), true);
        assert_eq!(vm.program_counter, 0);
    }

    #[test]
    fn test_vm_illegal() {
        let mut vm = VM::new();
        vm.command_buffer = vec![255];
        vm.init(2);
        let exit_code = vm.eager_step();
        assert_eq!(exit_code.is_ok(), false);
        assert_eq!(vm.program_counter, 1);
    }

    #[test]
    fn test_vm_fetch_instruction() {
        let mut vm = VM::new();
        vm.command_buffer = vec![0];
        let opcode = vm.fetch_instruction();
        assert_eq!(opcode.unwrap(), CRTOpCode::HALT);
    }

    #[test]
    fn test_vm_next_byte() {
        let mut vm = VM::new();
        vm.command_buffer = vec![8];
        let data = vm.get_next_byte();
        assert_eq!(data, 8);
    }

    #[test]
    fn test_vm_next_two_bytes() {
        let mut vm = VM::new();
        vm.command_buffer = vec![2, 7];
        let data = vm.get_next_two_bytes();
        assert_eq!(data, 519);
    }
}
