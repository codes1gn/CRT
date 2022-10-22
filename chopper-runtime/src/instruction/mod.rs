use nom::types::CompleteStr;
use raptors::prelude::OpCodeLike;
use serde::{Deserialize, Serialize};

use raptors::prelude::*;

#[cfg(any(feature = "mock", feature = "blas"))]
use rublas::prelude::*;

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum CRTOpCode {
    HALT, // 0
    LOAD, // 1

    ADDI32,      // 2
    SUBI32,      // 3
    MULI32,      // 4
    FLOORDIVI32, // 5

    // const literal command
    CONSTI32, // 6
    CONSTF32, // 7

    ADDF32,      // 8
    SUBF32,      // 9
    MULF32,      // 10
    DIVF32,      // 11
    CONSTTENSOR, // 12
    MATMULF32,   // 13

    SVALUETENSOR, // 14
    RNGTENSOR,    // 15

    // Unary 16
    EXPF32,
    RETV,
    RESHAPE,
    TRANSPOSE,
    NOOP,

    // ILLEGAL op always id at last index
    ILLEGAL, // rest
}

// TODO
// ===================    arithmetic ops
// add
// sub
// mul
// div
// rem
// fma
// abs f32
// neg f32
// ceil f32
// floor f32
// atan f32
// atan2 f32
// cos f32
// sin f32
// exp f32
// exp2 f32
// expm1 f32
// log f32
// log10 f32
// log1p f32
// log2 f32
// pow f32
// rsprt f32
// sprt f32
// tanh f32
// NOT i32
// AND i32
// OR i32
// XOR i32
// ========================    casting ops
// bc_i32tof32
// bc_f32toi32
// =====================      shift ops
// shl
// shr
// ======================     invoke ops
// invoke
impl From<u8> for CRTOpCode {
    fn from(v: u8) -> Self {
        match v {
            0 => {
                return CRTOpCode::HALT;
            }
            1 => {
                return CRTOpCode::LOAD;
            }
            2 => {
                return CRTOpCode::ADDI32;
            }
            3 => {
                return CRTOpCode::SUBI32;
            }
            4 => {
                return CRTOpCode::MULI32;
            }
            5 => {
                return CRTOpCode::FLOORDIVI32;
            }
            6 => {
                return CRTOpCode::CONSTI32;
            }
            7 => {
                return CRTOpCode::CONSTF32;
            }
            8 => {
                return CRTOpCode::ADDF32;
            }
            9 => {
                return CRTOpCode::SUBF32;
            }
            10 => {
                return CRTOpCode::MULF32;
            }
            11 => {
                return CRTOpCode::DIVF32;
            }
            12 => {
                return CRTOpCode::CONSTTENSOR;
            }
            13 => {
                return CRTOpCode::MATMULF32;
            }
            14 => {
                return CRTOpCode::SVALUETENSOR;
            }
            15 => {
                return CRTOpCode::RNGTENSOR;
            }
            16 => {
                return CRTOpCode::EXPF32;
            }
            17 => {
                return CRTOpCode::RETV;
            }
            18 => {
                return CRTOpCode::RESHAPE;
            }
            19 => {
                return CRTOpCode::TRANSPOSE;
            }
            20 => {
                return CRTOpCode::NOOP;
            }
            _ => {
                return CRTOpCode::ILLEGAL;
            }
        }
    }
}

impl OpCodeLike for CRTOpCode {}

// #[cfg(feature = "vulkan")]
impl CRTOpCode {
    pub fn to_kernel_query_entry(&self) -> String {
        match self {
            // i32 types
            CRTOpCode::ADDI32 | CRTOpCode::SUBI32 | CRTOpCode::MULI32 | CRTOpCode::FLOORDIVI32 => {
                String::from("binary_arithmetic_i32")
            }

            // f32 types
            // TODO(tianyu), this file specify the kernel code file name
            CRTOpCode::ADDF32 | CRTOpCode::SUBF32 | CRTOpCode::MULF32 | CRTOpCode::DIVF32 => {
                String::from("binary_arithmetic_f32")
            }

            CRTOpCode::MATMULF32 => String::from("matrix_multiple_f32"),

            _ => panic!("not support this op for dispatch kernel"),
        }
    }

    pub fn to_specialise_bits(&self) -> u32 {
        match self {
            // add spec data
            CRTOpCode::ADDI32 | CRTOpCode::ADDF32 => 0_u32,

            // sub spec data
            CRTOpCode::SUBI32 | CRTOpCode::SUBF32 => 1_u32,

            // sub spec data
            CRTOpCode::MULI32 | CRTOpCode::MULF32 => 2_u32,

            // floordiv
            CRTOpCode::FLOORDIVI32 | CRTOpCode::DIVF32 => 3_u32,

            // matrix-multiple
            CRTOpCode::MATMULF32 => 4_u32,

            // TODO(tianyu): change matmul opcode into add opcode to fake the compute
            // CRTOpCode::MATMULF32 => 0_u32,
            _ => panic!("unsupported opcode for specilising kernels"),
        }
    }
}

impl From<CompleteStr<'_>> for CRTOpCode {
    fn from(s: CompleteStr<'_>) -> Self {
        match s {
            CompleteStr("halt") => CRTOpCode::HALT,
            CompleteStr("return") => CRTOpCode::RETV,
            CompleteStr("load") => CRTOpCode::LOAD,
            CompleteStr("crt.add.i32") => CRTOpCode::ADDI32,
            CompleteStr("crt.sub.i32") => CRTOpCode::SUBI32,
            CompleteStr("crt.mul.i32") => CRTOpCode::MULI32,
            CompleteStr("crt.floordiv.i32") => CRTOpCode::FLOORDIVI32,
            CompleteStr("crt.literal.const.i32") => CRTOpCode::CONSTI32,
            CompleteStr("crt.literal.const.f32") => CRTOpCode::CONSTF32,
            CompleteStr("crt.literal.const.tensor") => CRTOpCode::CONSTTENSOR,
            CompleteStr("crt.helper.svalue.tensor") => CRTOpCode::SVALUETENSOR,
            CompleteStr("crt.helper.rng.tensor") => CRTOpCode::RNGTENSOR,
            CompleteStr("crt.add.f32") => CRTOpCode::ADDF32,
            CompleteStr("crt.sub.f32") => CRTOpCode::SUBF32,
            CompleteStr("crt.exp.f32") => CRTOpCode::EXPF32,
            CompleteStr("crt.reshape") => CRTOpCode::RESHAPE,
            CompleteStr("crt.transpose") => CRTOpCode::TRANSPOSE,
            CompleteStr("crt.noop") => CRTOpCode::NOOP,
            CompleteStr("crt.mul.f32") => CRTOpCode::MULF32,
            CompleteStr("crt.matmul.f32") => CRTOpCode::MATMULF32,
            CompleteStr("crt.div.f32") => CRTOpCode::DIVF32,
            _ => {
                panic!("unknown inst");
            }
        }
    }
}

// Conversions to plugin-opcode-types
// TODO rename OpCode to ChopperOpCode
// TODO refactor to dedicate mods
#[cfg(feature = "blas")]
impl From<CRTOpCode> for BlasOpCode {
    fn from(item: CRTOpCode) -> Self {
        match item {
            CRTOpCode::ADDF32 => BlasOpCode::AddF,
            CRTOpCode::SUBF32 => BlasOpCode::SubF,
            CRTOpCode::MULF32 => BlasOpCode::MulF,
            CRTOpCode::DIVF32 => BlasOpCode::DivF,
            CRTOpCode::ADDI32 => BlasOpCode::AddI,
            CRTOpCode::SUBI32 => BlasOpCode::SubI,
            CRTOpCode::MULI32 => BlasOpCode::MulI,
            CRTOpCode::FLOORDIVI32 => BlasOpCode::DivI,
            CRTOpCode::MATMULF32 => BlasOpCode::GemmF,
            _ => panic!("conversion from ChopperOpCode to BlasOpCode not registered"),
        }
    }
}

#[cfg(feature = "blas")]
impl From<BlasOpCode> for CRTOpCode {
    fn from(item: BlasOpCode) -> Self {
        match item {
            BlasOpCode::AddF => CRTOpCode::ADDF32,
            BlasOpCode::SubF => CRTOpCode::SUBF32,
            BlasOpCode::MulF => CRTOpCode::MULF32,
            BlasOpCode::DivF => CRTOpCode::DIVF32,
            BlasOpCode::AddI => CRTOpCode::ADDI32,
            BlasOpCode::SubI => CRTOpCode::SUBI32,
            BlasOpCode::MulI => CRTOpCode::MULI32,
            BlasOpCode::DivI => CRTOpCode::FLOORDIVI32,
            BlasOpCode::GemmF => CRTOpCode::MATMULF32,
            _ => panic!("conversion from BlasOpCode to ChopperOpCode not registered"),
        }
    }
}

#[cfg(feature = "mock")]
impl From<CRTOpCode> for MockOpCode {
    fn from(item: CRTOpCode) -> Self {
        match item {
            CRTOpCode::EXPF32 => MockOpCode::ExpOp,
            CRTOpCode::RESHAPE => MockOpCode::ReshapeOp,
            CRTOpCode::ADDF32 => MockOpCode::AddOp,
            CRTOpCode::SUBF32 => MockOpCode::SubOp,
            CRTOpCode::MULF32 => MockOpCode::MulOp,
            CRTOpCode::DIVF32 => MockOpCode::DivOp,
            CRTOpCode::ADDI32 => MockOpCode::AddOp,
            CRTOpCode::SUBI32 => MockOpCode::SubOp,
            CRTOpCode::MULI32 => MockOpCode::MulOp,
            CRTOpCode::FLOORDIVI32 => MockOpCode::DivOp,
            CRTOpCode::MATMULF32 => MockOpCode::MatmulOp,
            _ => panic!("conversion from ChopperOpCode to MockOpCode not registered"),
        }
    }
}

#[cfg(feature = "blas")]
impl From<MockOpCode> for CRTOpCode {
    fn from(item: MockOpCode) -> Self {
        match item {
            MockOpCode::ExpOp => CRTOpCode::EXPF32,
            MockOpCode::ReshapeOp => CRTOpCode::RESHAPE,
            MockOpCode::AddOp => CRTOpCode::ADDF32,
            MockOpCode::SubOp => CRTOpCode::SUBF32,
            MockOpCode::MulOp => CRTOpCode::MULF32,
            MockOpCode::DivOp => CRTOpCode::DIVF32,
            // MockOpCode::SinOp => CRTOpCode::SINF32,
            // MockOpCode::ExpOp => CRTOpCode::EXPF32,
            // pub enum MockOpCode {
            //     IdentityOp,
            //     AddOp,
            //     SubOp,
            //     MulOp,
            //     DivOp,
            //     ConvOp,
            //     ExpOp,
            //     MatmulOp,
            //     SinOp,
            // }
            MockOpCode::MatmulOp => CRTOpCode::MATMULF32,
            _ => panic!("conversion from MockOpCode to ChopperOpCode not registered"),
        }
    }
}

// Inst types definition
pub struct Instruction {
    opcode: CRTOpCode,
}

// Note
impl Instruction {
    pub fn new(opcode: CRTOpCode) -> Instruction {
        Instruction { opcode: opcode }
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_opcodes() {
        let opcode = CRTOpCode::HALT;
        assert_eq!(opcode, CRTOpCode::HALT);

        let opcode = CRTOpCode::LOAD;
        assert_eq!(opcode, CRTOpCode::LOAD);

        let opcode = CRTOpCode::ADDI32;
        assert_eq!(opcode, CRTOpCode::ADDI32);

        let opcode = CRTOpCode::SUBI32;
        assert_eq!(opcode, CRTOpCode::SUBI32);

        let opcode = CRTOpCode::MULI32;
        assert_eq!(opcode, CRTOpCode::MULI32);

        let opcode = CRTOpCode::FLOORDIVI32;
        assert_eq!(opcode, CRTOpCode::FLOORDIVI32);

        let opcode = CRTOpCode::CONSTI32;
        assert_eq!(opcode, CRTOpCode::CONSTI32);

        let opcode = CRTOpCode::CONSTF32;
        assert_eq!(opcode, CRTOpCode::CONSTF32);

        let opcode = CRTOpCode::CONSTTENSOR;
        assert_eq!(opcode, CRTOpCode::CONSTTENSOR);
    }

    #[test]
    fn test_create_opcode() {
        let opcode = CRTOpCode::ILLEGAL;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, CRTOpCode::ILLEGAL);

        let opcode = CRTOpCode::CONSTF32;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, CRTOpCode::CONSTF32);

        let opcode = CRTOpCode::CONSTI32;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, CRTOpCode::CONSTI32);

        let opcode = CRTOpCode::CONSTTENSOR;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, CRTOpCode::CONSTTENSOR);

        let opcode = CRTOpCode::MATMULF32;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, CRTOpCode::MATMULF32);
    }
}
