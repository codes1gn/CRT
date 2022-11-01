// external crates
use nom::types::CompleteStr;
use nom::*;

// submods
pub mod assembler_base;
pub mod parse_helper;
pub mod parse_instruction;
pub mod parse_literal;
pub mod parse_module;
pub mod parse_opcode;
pub mod parse_operand;
pub mod parse_type;

use assembler_base::*;
use parse_instruction::*;
use parse_module::*;

// interface for bytecode parsing
named!(pub parse_bytecode<CompleteStr, Program>,
    do_parse!(
        program: parse_raw_program >> (
            program
        )
    )
);

// TODO local program + remote programs = compound-bytecode
// TODO impl remote programs with rpc support
named!(pub parse_raw_program<CompleteStr, Program>,
    do_parse!(
        modules: many1!(
            alt!(
                parse_module |
                parse_phantom_module
            )
        ) >> (
            Program {
                mods: modules
            }
        )
    )
);

#[cfg(feature = "mock")]
named!(pub parse_mock_program<CompleteStr, Program>,
    do_parse!(
        modules: many1!(
            alt!(
                parse_mock_module
            )
        ) >> (
            Program {
                mods: modules
            }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "mock")]
    #[test]
    fn test_parse_mock_program() {
        let bytecode = "
        %0 = crt.maxpool %100 : (tensor<1x1x28x28xf32>) -> tensor<1x1x14x14xf32>
        %1 = crt.constant : () -> tensor<2xi64>
        %3 = crt.constant : () -> tensor<128x196xf32>
        %4 = crt.constant : () -> tensor<128xf32>
        %7 = crt.constant : () -> tensor<10x128xf32>
        %8 = crt.constant : () -> tensor<10xf32>
        %6 = crt.relu %5 : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %10 = crt.softmax %9 : (tensor<1x10xf32>) -> tensor<1x10xf32>
        return %10 : tensor<1x10xf32>
        ";
        let full_result = parse_mock_module(CompleteStr(bytecode));
        // assert_eq!(full_result.is_ok(), true);
        let (_remain, _parsed) = full_result.unwrap();
        println!("{:?}", _remain);
        println!("{:?}", _parsed);
        assert_eq!(_remain.is_empty(), true);
    }

    #[test]
    fn test_parse_multiple_module_program() {
        let bytecode = "
        %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
        %5 = crt.add.f32! %3, %4: f32

        phantom.block <dev:#1> {
            %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
            %1 = crt.reshape! %0, [4096 1024]
            %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %3 = crt.matmul.f32! %1, %2 : f32
            %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %5 = crt.add.f32! %3, %4: f32
            %6 = crt.reshape! %5, [32 128 16 64]
            %101 = crt.transpose! %6, [32 16 128 64]
        }

        phantom.block <dev:#2> {
            %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
            %1 = crt.reshape! %0, [4096 1024]
            %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %3 = crt.matmul.f32! %1, %2 : f32
            %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %5 = crt.add.f32! %3, %4: f32
            %6 = crt.reshape! %5, [32 128 16 64]
            %101 = crt.transpose! %6, [32 16 128 64]
        }

        %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
        %5 = crt.add.f32! %3, %4: f32
        ";
        let full_result = parse_raw_program(CompleteStr(bytecode));
        // assert_eq!(full_result.is_ok(), true);
        let (_remain, _parsed) = full_result.unwrap();
        println!("{:?}", _remain);
        println!("{:?}", _parsed);
        assert_eq!(_remain.is_empty(), true);
        assert_eq!(_parsed.dev_at_mod_idx(0), None);
        assert_eq!(_parsed.dev_at_mod_idx(1), Some(1));
        assert_eq!(_parsed.dev_at_mod_idx(2), Some(2));
        assert_eq!(_parsed.dev_at_mod_idx(3), None);
    }
}
