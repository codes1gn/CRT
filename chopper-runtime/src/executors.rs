use std::{borrow::Cow, collections::HashMap, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use raptors::prelude::*;

#[cfg(any(feature = "mock", feature = "blas"))]
use rublas::prelude::*;

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
use crate::vkgpu_executor::*;

#[derive(Debug)]
pub enum ActExecutorTypes {
    VkGPUExecutor(VkGPUExecutor),
    MockExecutor(MockExecutor),

    #[cfg(all(feature = "blas"))]
    BlasExecutor(BlasExecutor),
}

impl ExecutorLike for ActExecutorTypes {
    type TensorType = ActTensorTypes;
    type OpCodeType = instruction::OpCode;
    fn new_with_typeid(typeid: usize) -> ActExecutorTypes {
        match typeid {
            0 => ActExecutorTypes::MockExecutor(MockExecutor::new()),
            1 => ActExecutorTypes::VkGPUExecutor(VkGPUExecutor::new()),

            #[cfg(all(feature = "blas"))]
            2 => ActExecutorTypes::BlasExecutor(BlasExecutor::new()),

            _ => panic!("not registered backend typeid"),
        }
    }

    fn init(&mut self) {
        match self {
            ActExecutorTypes::MockExecutor(_) => {
                println!("no action in init");
            }
            ActExecutorTypes::VkGPUExecutor(ref mut e) => {
                e.raw_init();
            }

            #[cfg(all(feature = "blas"))]
            ActExecutorTypes::BlasExecutor(_) => {
                println!("no action in init");
            }

            _ => panic!("not registered backend typeid"),
        }
    }

    fn mock_compute(&mut self, in_tensor: Self::TensorType) -> Self::TensorType {
        // println!("============ on computing =============");
        panic!("cannot compute");
        in_tensor
    }

    fn unary_compute(
        &mut self,
        op: Self::OpCodeType,
        in_tensor: Self::TensorType,
    ) -> Self::TensorType {
        // debug!("============ on computing unary =============");
        panic!("cannot compute");
        in_tensor
    }

    fn binary_compute(
        &mut self,
        op: Self::OpCodeType,
        lhs_tensor: Self::TensorType,
        rhs_tensor: Self::TensorType,
    ) -> Self::TensorType {
        // debug!("============ on computing binary =============");
        match self {
            ActExecutorTypes::MockExecutor(ref mut _executor) => {
                _executor.mock_binary::<Self::TensorType>(lhs_tensor, rhs_tensor)
                // TODO use pattern match on matching tensortypes, rather than call as generic
                // WIP match lhs_tensor {
                // WIP     // TODO WIP make this MockTensor
                // WIP     ActTensorTypes::F32Tensor { data } => {
                // WIP         let lhs_data = data;
                // WIP         match rhs_tensor {
                // WIP             ActTensorTypes::F32Tensor { data } => {
                // WIP                 let rhs_data = data;
                // WIP                 return ActTensorTypes::MockTensor {
                // WIP                     data: _executor.mock_binary::<Self::TensorType>(lhs_data, rhs_data),
                // WIP                     // WIP data: _executor.binary_compute(op, lhs_data, rhs_data),
                // WIP                 };
                // WIP             }
                // WIP             _ => panic!("lhs and rhs type mismatch"),
                // WIP         }
                // WIP     }
                // WIP     ActTensorTypes::MockTensor => panic!("TODO WIP, CRT use TensorView, make this changable"),
                // WIP     _ => panic!("dtype not compatible, exp_executor <type: MockTensor> {:#?}", lhs_tensor),
                // WIP }
            }
            ActExecutorTypes::VkGPUExecutor(ref mut _executor) => {
                match lhs_tensor {
                    ActTensorTypes::F32Tensor { data } => {
                        let lhs_data = data;
                        match rhs_tensor {
                            ActTensorTypes::F32Tensor { data } => {
                                let rhs_data = data;
                                return ActTensorTypes::F32Tensor {
                                    data: _executor.binary_compute_f32(op, lhs_data, rhs_data),
                                };
                            }
                            _ => panic!("dtype mismatch"),
                        }
                    }
                    ActTensorTypes::I32Tensor { data } => {
                        let lhs_data = data;
                        match rhs_tensor {
                            ActTensorTypes::I32Tensor { data } => {
                                let rhs_data = data;
                                return ActTensorTypes::I32Tensor {
                                    data: _executor.binary_compute_i32(op, lhs_data, rhs_data),
                                };
                            }
                            _ => panic!("dtype mismatch"),
                        }
                    }
                    _ => panic!("dtype-comp not implemented"),
                };
            }
            #[cfg(all(feature = "blas"))]
            ActExecutorTypes::BlasExecutor(ref mut _executor) => {
                match lhs_tensor {
                    ActTensorTypes::F32Tensor { data } => {
                        let lhs_data = data.into();
                        match rhs_tensor {
                            ActTensorTypes::F32Tensor { data } => {
                                let rhs_data = data.into();
                                let out = ActTensorTypes::F32Tensor {
                                    // TODO tadd to be replace into binary and unary
                                    // op to be handled
                                    data: _executor
                                        .binary_compute_owned(op.into(), lhs_data, rhs_data)
                                        .into(),
                                };
                                // println!("{:#?}", out);
                                return out;
                            }
                            _ => panic!("dtype mismatch"),
                        }
                    }
                    ActTensorTypes::I32Tensor { data } => {
                        let lhs_data = data.into();
                        match rhs_tensor {
                            ActTensorTypes::I32Tensor { data } => {
                                let rhs_data = data.into();
                                let out = ActTensorTypes::I32Tensor {
                                    // TODO tadd to be replace into binary and unary
                                    // op to be handled
                                    data: _executor
                                        .binary_compute_owned(op.into(), lhs_data, rhs_data)
                                        .into(),
                                };
                                // println!("{:#?}", out);
                                return out;
                            }
                            _ => panic!("dtype mismatch"),
                        }
                    }
                    _ => panic!("dtype-comp not implemented"),
                };
            }
            _ => panic!("not registered backend typeid"),
        }
    }
}

// TODO rename OpCode to ChopperOpCode
// TODO refactor to dedicate mods
#[cfg(feature = "blas")]
impl From<instruction::OpCode> for BlasOpCode {
    fn from(item: instruction::OpCode) -> Self {
        match item {
            instruction::OpCode::ADDF32 => BlasOpCode::AddF,
            instruction::OpCode::SUBF32 => BlasOpCode::SubF,
            instruction::OpCode::MULF32 => BlasOpCode::MulF,
            instruction::OpCode::DIVF32 => BlasOpCode::DivF,
            instruction::OpCode::ADDI32 => BlasOpCode::AddI,
            instruction::OpCode::SUBI32 => BlasOpCode::SubI,
            instruction::OpCode::MULI32 => BlasOpCode::MulI,
            instruction::OpCode::FLOORDIVI32 => BlasOpCode::DivI,
            instruction::OpCode::MATMULF32 => BlasOpCode::GemmF,
            _ => panic!("conversion from ChopperOpCode to BlasOpCode not registered"),
        }
    }
}

#[cfg(feature = "blas")]
impl From<BlasOpCode> for instruction::OpCode {
    fn from(item: BlasOpCode) -> Self {
        match item {
            BlasOpCode::AddF => instruction::OpCode::ADDF32,
            BlasOpCode::SubF => instruction::OpCode::SUBF32,
            BlasOpCode::MulF => instruction::OpCode::MULF32,
            BlasOpCode::DivF => instruction::OpCode::DIVF32,
            BlasOpCode::AddI => instruction::OpCode::ADDI32,
            BlasOpCode::SubI => instruction::OpCode::SUBI32,
            BlasOpCode::MulI => instruction::OpCode::MULI32,
            BlasOpCode::DivI => instruction::OpCode::FLOORDIVI32,
            BlasOpCode::GemmF => instruction::OpCode::MATMULF32,
            _ => panic!("conversion from BlasOpCode to ChopperOpCode not registered"),
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;
}
