use nom::types::CompleteStr;
use nom::*;

use serde::{Deserialize, Serialize};

use crate::base::*;
use crate::instruction::CRTOpCode;

// enum type can accept struct-like value.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Token {
    BytecodeOpCode {
        code: CRTOpCode,
    },
    Variable {
        symbol: u8,
    },
    // TODO maybe support Variable { symbol: String },
    I32Literal {
        value: i32,
    },
    F32Literal {
        value: f32,
    },
    Tensor {
        raw_data: Vec<f32>,
        shape: Vec<usize>,
    },
    TensorType {
        dtype: ElementType,
        shape: Vec<usize>,
    },
    UninitTensor {
        data_generator: f32,
        shape: Vec<usize>,
    },
    // distribution => u8 : 0 -> uniform; 1 -> normal
    // for uniform, arg1 => min, arg2 => max
    // for normal, arg1 => mean, arg2 => std
    // TODO use above setting, currently only support -1.0 -> 1.0 uniform and 0.0 - 1.0 normal
    UninitRNGTensor {
        distribution: u8,
        shape: Vec<usize>,
    },
    DType {
        element_type: ElementType,
    },
    Shape {
        raw_shape: Vec<usize>,
    },
}

#[cfg(not(feature = "phantom"))]
#[derive(Debug, PartialEq, Clone)]
pub struct AsmInstruction {
    pub(crate) opcode: Token,
    pub(crate) operand1: Option<Token>,
    pub(crate) operand2: Option<Token>,
    pub(crate) operand3: Option<Token>,
}

// The abstract struct for asm inst.
#[cfg(feature = "phantom")]
#[derive(Debug, PartialEq, Clone)]
pub struct AsmInstruction {
    pub(crate) opcode: Token,
    pub(crate) operand1: Option<Token>,
    pub(crate) operand2: Option<Token>,
    pub(crate) operand3: Option<Token>,
    pub(crate) operand4: Option<Token>,
    pub(crate) operand2_type: Option<Token>,
    pub(crate) operand3_type: Option<Token>,
    pub(crate) operand4_type: Option<Token>,
    pub(crate) result_type: Option<Token>,
}

// impl a function that can throw the asminstruction into a Vec<u8> format
impl AsmInstruction {
    // getters

    // serialise function from AsmInstruction struct to Vec<u8> that compatible to command buffer
    pub fn to_bytes(self: &Self) -> Vec<u8> {
        let mut results = vec![];
        // have to allow for copy and clone for opcode, since we need to apply as u8 on it, rather
        // than on the ref
        //
        // match opcode
        match &self.opcode {
            Token::BytecodeOpCode { code } => {
                results.push(*code as u8);
            }
            _ => {
                panic!("Unsuported opcode found");
            }
        }

        // match operand1
        match &self.operand1 {
            Some(t) => match t {
                Token::Variable { symbol } => {
                    results.push(*symbol);
                }
                Token::I32Literal { value } => {
                    // convert i32 into 4 of bytes in little endian order
                    // push it into cmd_buffer
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::F32Literal { value } => {
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::UninitTensor {
                    data_generator,
                    shape,
                } => {
                    // push data_generator
                    let values = data_generator.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::UninitRNGTensor {
                    distribution,
                    shape,
                } => {
                    // push distribution
                    results.push(*distribution);
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::Tensor { raw_data, shape } => {
                    let data_bytes: Vec<u8> = bincode::serialize(&raw_data).unwrap();
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let data_len = data_bytes.len() as u16;
                    let data_len_bytes = data_len.to_le_bytes();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    // println!("{:?}", data_len);
                    // println!("{:?}", shape_bytes);
                    // println!("{:?}", shape_len);
                    // assert_eq!(0, 1);
                    // println!("{:?}", data_len_bytes.len());
                    // assert_eq!(0, 1);
                    for _data_len in data_len_bytes {
                        results.push(_data_len);
                    }
                    for _data in data_bytes {
                        results.push(_data)
                    }
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }

        // match operand2
        // TODO handle too long digits of i32
        match &self.operand2 {
            Some(t) => match t {
                Token::Variable { symbol } => {
                    results.push(*symbol);
                }
                Token::I32Literal { value } => {
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::F32Literal { value } => {
                    // println!("{:?}", value);
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::UninitTensor {
                    data_generator,
                    shape,
                } => {
                    // push data_generator
                    let values = data_generator.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::UninitRNGTensor {
                    distribution,
                    shape,
                } => {
                    // push distribution
                    results.push(*distribution);
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::Tensor { raw_data, shape } => {
                    let data_bytes: Vec<u8> = bincode::serialize(&raw_data).unwrap();
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let data_len = data_bytes.len() as u16;
                    let data_len_bytes = data_len.to_le_bytes();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    // println!("{:?}", data_len);
                    println!("encoding data len {:?}", data_len_bytes);
                    println!("encoding data bytes {:?}", data_bytes);
                    println!("encoding shape len {:?}", shape_len_bytes);
                    println!("encoding shape bytes {:?}", shape_bytes);
                    // println!("{:?}", shape_len);
                    // assert_eq!(0, 1);
                    // println!("{:?}", data_len_bytes.len());
                    // assert_eq!(0, 1);
                    for _data_len in data_len_bytes {
                        results.push(_data_len);
                    }
                    for _data in data_bytes {
                        results.push(_data)
                    }
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }

        // match operand3
        match &self.operand3 {
            Some(t) => match t {
                Token::Variable { symbol } => {
                    results.push(*symbol);
                }
                Token::I32Literal { value } => {
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::F32Literal { value } => {
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::Shape { raw_shape } => {
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&raw_shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                // used for paper
                #[cfg(feature = "phantom")]
                Token::TensorType { dtype, shape } => {
                    let dtype_code: u8 = dtype.into();
                    results.push(dtype_code);

                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::UninitTensor {
                    data_generator,
                    shape,
                } => {
                    // push data_generator
                    let values = data_generator.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::UninitRNGTensor {
                    distribution,
                    shape,
                } => {
                    // push distribution
                    results.push(*distribution);
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::Tensor { raw_data, shape } => {
                    let data_bytes: Vec<u8> = bincode::serialize(&raw_data).unwrap();
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let data_len = data_bytes.len() as u16;
                    let data_len_bytes = data_len.to_le_bytes();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    // println!("{:?}", data_len);
                    // println!("{:?}", shape_bytes);
                    // println!("{:?}", shape_len);
                    // assert_eq!(0, 1);
                    // println!("{:?}", data_len_bytes.len());
                    // assert_eq!(0, 1);
                    for _data_len in data_len_bytes {
                        results.push(_data_len);
                    }
                    for _data in data_bytes {
                        results.push(_data)
                    }
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }
        // ANCHOR use this in phantom branch
        #[cfg(feature = "phantom")]
        match &self.operand4 {
            Some(t) => match t {
                Token::Variable { symbol } => {
                    results.push(*symbol);
                }
                Token::I32Literal { value } => {
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::F32Literal { value } => {
                    let values = value.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                }
                Token::Shape { raw_shape } => {
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&raw_shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                // used for paper
                #[cfg(feature = "phantom")]
                Token::TensorType { dtype, shape } => {
                    let dtype_code: u8 = dtype.into();
                    results.push(dtype_code);

                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::UninitTensor {
                    data_generator,
                    shape,
                } => {
                    // push data_generator
                    let values = data_generator.to_le_bytes();
                    for _value in values {
                        results.push(_value);
                    }
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::UninitRNGTensor {
                    distribution,
                    shape,
                } => {
                    // push distribution
                    results.push(*distribution);
                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                Token::Tensor { raw_data, shape } => {
                    let data_bytes: Vec<u8> = bincode::serialize(&raw_data).unwrap();
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let data_len = data_bytes.len() as u16;
                    let data_len_bytes = data_len.to_le_bytes();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    // println!("{:?}", data_len);
                    // println!("{:?}", shape_bytes);
                    // println!("{:?}", shape_len);
                    // assert_eq!(0, 1);
                    // println!("{:?}", data_len_bytes.len());
                    // assert_eq!(0, 1);
                    for _data_len in data_len_bytes {
                        results.push(_data_len);
                    }
                    for _data in data_bytes {
                        results.push(_data)
                    }
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }
        #[cfg(feature = "phantom")]
        match &self.operand2_type {
            Some(t) => match t {
                // used for paper
                Token::TensorType { dtype, shape } => {
                    let dtype_code: u8 = dtype.into();
                    results.push(dtype_code);

                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }

        #[cfg(feature = "phantom")]
        match &self.operand3_type {
            Some(t) => match t {
                // used for paper
                Token::TensorType { dtype, shape } => {
                    let dtype_code: u8 = dtype.into();
                    results.push(dtype_code);

                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }

        #[cfg(feature = "phantom")]
        match &self.operand4_type {
            Some(t) => match t {
                // used for paper
                Token::TensorType { dtype, shape } => {
                    let dtype_code: u8 = dtype.into();
                    results.push(dtype_code);

                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }

        #[cfg(feature = "phantom")]
        match &self.result_type {
            Some(t) => match t {
                // used for paper
                Token::TensorType { dtype, shape } => {
                    let dtype_code: u8 = dtype.into();
                    results.push(dtype_code);

                    // push shape
                    let shape_bytes: Vec<u8> = bincode::serialize(&shape).unwrap();
                    let shape_len = shape_bytes.len() as u16;
                    let shape_len_bytes = shape_len.to_le_bytes();
                    for _shape_len in shape_len_bytes {
                        results.push(_shape_len);
                    }
                    for _shape in shape_bytes {
                        results.push(_shape)
                    }
                }
                _ => {
                    panic!("register or literal/operand only");
                }
            },
            None => {} // do nothing if this operand is empty
        }

        return results;
    }
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub(crate) mods: Vec<Module>,
}

impl Program {
    pub(crate) fn dev_at_mod_idx(&self, pos: usize) -> Option<u8> {
        self.mods[pos].dev()
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut mod_bytes = vec![];
        for _mod in &self.mods {
            mod_bytes.append(&mut _mod.to_bytes());
        }
        return mod_bytes;
    }
}

#[derive(Debug, PartialEq)]
pub struct Module {
    pub(crate) dev_at: Option<u8>,
    pub(crate) instructions: Vec<AsmInstruction>,
}

// TODO move prase_program to submod, defines the Trait interface in mod.rs and pub it to the
// outside
impl Module {
    pub fn dev(&self) -> Option<u8> {
        self.dev_at
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut inst_bytes = vec![];
        // TODO add dev_at inst
        inst_bytes.push(CRTOpCode::DEVAT as u8);
        match self.dev_at {
            Some(idx) => {
                inst_bytes.push(idx);
            }
            None => {
                // dev id should be 0-N, -1 means none-dev
                inst_bytes.push(u8::MAX as u8);
            }
        }
        for inst in &self.instructions {
            inst_bytes.append(&mut inst.to_bytes());
        }
        return inst_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
