// external crates
use nom::types::CompleteStr;
use nom::*;

// mods from local crate
use crate::instruction::CRTOpCode;

// use assembler_base::*;
use crate::assembler::assembler_base::*;
use crate::assembler::parse_helper::*;
use crate::assembler::parse_literal::*;
use crate::assembler::parse_opcode::*;
use crate::assembler::parse_operand::*;
use crate::assembler::parse_type::*;

named!(pub parse_instruction<CompleteStr, AsmInstruction>,
    do_parse!(
        _inst: alt!(
            parse_halt | parse_return | parse_binary_assignment | parse_unary_assignment | parse_unary_assignment_with_shape_argument | parse_comment
        ) >>
        opt!(multispace) >>
        (
            _inst
        )
    )
);

#[cfg(feature = "mock")]
named!(pub parse_mock_instruction<CompleteStr, AsmInstruction>,
    do_parse!(
        opt!(multispace) >>
        _inst: alt!(
            parse_mock_zenary_instruction | parse_mock_binary_instruction | parse_mock_unary_instruction | parse_comment | parse_mock_return
        ) >>
        opt!(multispace) >>
        (
            _inst
        )
    )
);

// halt
named!(parse_halt<CompleteStr, AsmInstruction>,
    do_parse!(
        _opcode: alt!(
            tag!("halt")
        ) >>
        // must use multispace, since we have to identify change lines of halt itself.
        opt!(multispace) >>
        (
            AsmInstruction {
                opcode: Token::BytecodeOpCode { code: CRTOpCode::from(_opcode) },
                operand1: None,
                operand2: None,
                operand3: None,
            }
        )
    )
);

// "// some strings for comment\n"
// TODO impl multiline comment parsing
// take /*, then take_until */
named!(parse_comment<CompleteStr, AsmInstruction>,
    do_parse!(
        tag!("//") >>
        alt!(
            take_until!("\n") | take_until!("\r\n")
        ) >>
        (
            AsmInstruction {
                opcode: Token::BytecodeOpCode { code: CRTOpCode::from(CompleteStr("crt.noop")) },
                operand1: None,
                operand2: None,
                operand3: None,
            }
        )
    )
);

#[cfg(feature = "mock")]
named!(parse_mock_return<CompleteStr, AsmInstruction>,
    do_parse!(
        _opcode: alt!(
            tag!("return")
        ) >>
        opt!(multispace) >>
        // TODO runtime function is functional, no side
        //  effect, must clear all results without returns
        _operand1: parse_operand >>
        space0 >>
        tag!(":") >>
        space0 >>
        _dtype: parse_ranked_tensor_type >>
        // must use multispace, since we have to identify change lines of halt itself.
        opt!(multispace) >>
        (
            AsmInstruction {
                opcode: Token::BytecodeOpCode { code: CRTOpCode::from(_opcode) },
                operand1: Some(_operand1),
                operand2: None,
                operand3: None,
            }
        )
    )
);

// return %1
named!(parse_return<CompleteStr, AsmInstruction>,
    do_parse!(
        _opcode: alt!(
            tag!("return")
        ) >>
        opt!(multispace) >>
        // TODO runtime function is functional, no side
        //  effect, must clear all results without returns
        _operand1: parse_operand >>
        // must use multispace, since we have to identify change lines of halt itself.
        opt!(multispace) >>
        (
            AsmInstruction {
                opcode: Token::BytecodeOpCode { code: CRTOpCode::from(_opcode) },
                operand1: Some(_operand1),
                operand2: None,
                operand3: None,
            }
        )
    )
);

// return %2, %1
// TODO multi-ret-values support
// named!(parse_return_pair<CompleteStr, AsmInstruction>,
//     do_parse!(
//         _opcode: alt!(
//             tag!("return")
//         ) >>
//         opt!(multispace) >>
//         // TODO runtime function is functional, no side
//         //  effect, must clear all results without returns
//         _operand1: parse_operand >>
//         tag!(", ") >>
//         _operand2: parse_operand >>
//         // must use multispace, since we have to identify change lines of halt itself.
//         opt!(multispace) >>
//         (
//             AsmInstruction {
//                 opcode: Token::BytecodeOpCode { code: CRTOpCode::from(_opcode) },
//                 operand1: Some(_operand1),
//                 operand2: Some(_operand1),
//                 operand3: None,
//             }
//         )
//     )
// );
//
//
// ANCHOR mock
#[cfg(feature = "mock")]
named!(
    parse_mock_binary_instruction<CompleteStr, AsmInstruction>,
    do_parse!(
        _result: parse_operand >>
        tag!("= ") >>
        _opcode: parse_mock_opcode >>
        _operand_lhs: parse_operand >>
        tag!(", ") >>
        _operand_rhs: parse_operand >>
        tag!(": ") >>
        // ANCHOR add functiontype parse here, binary op ignores shape, currently make shape rule by runtime
        // status nor this static type info
        _dtype: parse_function_type >>
        (
            AsmInstruction {
                opcode: _opcode,
                operand1: Some(_result),
                operand2: Some(_operand_lhs),
                operand3: Some(_operand_rhs),
            }
        )
    )
);

#[cfg(feature = "mock")]
named!(
    parse_mock_unary_instruction<CompleteStr, AsmInstruction>,
    do_parse!(
        out_operand: parse_operand >>
        tag!("=") >>
        _s1: space0 >>
        opcode: parse_mock_opcode >>
        in_operand: parse_operand >>
        tag!(": ") >>
        _dtype: parse_function_type >>
        (
            AsmInstruction {
                opcode: opcode,
                operand1: Some(out_operand),
                operand2: Some(in_operand),
                operand3: None,

            }
        )
    )
);

#[cfg(feature = "mock")]
named!(
    parse_mock_zenary_instruction<CompleteStr, AsmInstruction>,
    do_parse!(
        out_operand: parse_operand >>
        tag!("=") >>
        _s1: space0 >>
        opcode: parse_mock_opcode >>
        space0 >>
        tag!(": ") >>
        _dtype: parse_function_type >>
        (
            // Token::UninitTensor { data_generator: 0f32, shape: _shape }
            // Token::TensorType {
            //     dtype: ElementType::from(dtype_token),
            //     shape: ranked_shape,
            // }
            match _dtype {
                Token::TensorType { dtype, shape } => {
                    AsmInstruction {
                        opcode: opcode,
                        operand1: Some(out_operand),
                        operand2: Some(Token::UninitTensor {
                            // hardcode for now
                            data_generator: 0f32,
                            shape: shape,
                        }),
                        operand3: None,
                    }
                },
                _ => panic!("expect token::tensor-type"),
            }
        )
    )
);

// binary-assignment ::= out-operand opcode lhs-operand rhs-operand
// lhs-operand ::= operand | numeric-literal
// rhs-operand ::= operand | numeric-literal
// %86 = crt.add %82, %85 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
named!(
    parse_binary_assignment<CompleteStr, AsmInstruction>,
    do_parse!(
        _result: parse_operand >>
        tag!("= ") >>
        _opcode: parse_opcode >>
        _operand_lhs: parse_operand >>
        tag!(", ") >>
        _operand_rhs: parse_operand >>
        tag!(": ") >>
        // ANCHOR add functiontype parse here, binary op ignores shape, currently make shape rule by runtime
        // status nor this static type info
        _dtype: alt!(parse_type | parse_function_type) >>
        (
            AsmInstruction {
                opcode: _opcode,
                operand1: Some(_result),
                operand2: Some(_operand_lhs),
                operand3: Some(_operand_rhs),
            }
        )
    )
);

// unary-assignment ::= out-operand = opcode in-operand
// in-operand ::= operand | numeric-literal
named!(
    parse_unary_assignment<CompleteStr, AsmInstruction>,
    do_parse!(
        out_operand: parse_operand >>
        tag!("=") >>
        _s1: space0 >>
        opcode: parse_opcode >>
        in_operand: alt!(
            parse_operand_with_type
            | parse_float_literal_with_type
            | parse_integer_literal_with_type
            | parse_tensor_literal
            | parse_helper_zeros
            | parse_helper_ones
            | parse_helper_uniform
            | parse_helper_normal
        ) >>
        (
            AsmInstruction {
                opcode: opcode,
                operand1: Some(out_operand),
                operand2: Some(in_operand),
                operand3: None,

            }
        )
    )
);

// unary-assignment-with-indexed-arg ::= out-operand = opcode in-operand, indexed-arg
// "%0 = crt.reshape! %1, [2 3]\n",
named!(
    parse_unary_assignment_with_shape_argument<CompleteStr, AsmInstruction>,
    do_parse!(
        out_operand: parse_operand >>
        tag!("=") >>
        _s1: space0 >>
        _opcode: parse_opcode >>
        in_operand: parse_operand >>
        tag!(",") >>
        _shape: parse_shape >>
        opt!(multispace) >>
        (
            AsmInstruction {
                opcode: _opcode,
                operand1: Some(out_operand),
                operand2: Some(in_operand),
                operand3: Some(_shape),
            }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_halt_from_bytecode() {
        let result = parse_instruction(CompleteStr("halt\n"));
        assert_eq!(result.is_ok(), true);
        let _result = result.unwrap();
        assert_eq!(_result.0.is_empty(), true);
        assert_eq!(
            _result.1.opcode,
            Token::BytecodeOpCode {
                code: CRTOpCode::HALT
            }
        );
    }

    // tests that covers parse_halt
    #[test]
    fn test_parse_halt() {
        let result = parse_instruction(CompleteStr("halt\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![0])
    }

    // tests that covers parse_halt
    #[test]
    fn test_parse_return() {
        let result = parse_instruction(CompleteStr("return %1\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![17, 1])
    }

    // tests that covers parse_binary
    #[test]
    fn test_parse_assignment_add() {
        let result = parse_instruction(CompleteStr("%0 = crt.add.i32! %2, %3 : i32\n"));
        assert_eq!(result.is_ok(), true);
        println!("{:?}", result);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![2, 0, 2, 3])
    }

    #[test]
    fn test_parse_assignment_sub() {
        let result = parse_instruction(CompleteStr("%0 = crt.sub.i32! %2, %3 : i32\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![3, 0, 2, 3])
    }

    #[test]
    fn test_parse_assignment_mul() {
        let result = parse_instruction(CompleteStr("%1 = crt.mul.i32! %2, %3 : i32\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![4, 1, 2, 3])
    }

    #[test]
    fn test_parse_assignment_floordiv() {
        let result = parse_instruction(CompleteStr("%3 = crt.floordiv.i32! %2, %0 : i32\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![5, 3, 2, 0])
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_parse_mock_exp() {
        let result = parse_mock_unary_instruction(CompleteStr(
            "%3 = crt.exp %2 : (tensor<1x128xf32>) -> tensor<1x128xf32>",
        ));
        assert_eq!(result.is_ok(), true);
        let (_remain, _bytes_result) = result.unwrap();
        println!("{:?}", _remain);
        assert_eq!(_bytes_result.to_bytes(), vec![16, 3, 2])
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_parse_mock_maxpool() {
        let result = parse_mock_unary_instruction(CompleteStr(
            "%0 = crt.maxpool %100 : (tensor<1x1x28x28xf32>) -> tensor<1x1x14x14xf32>",
        ));
        // assert_eq!(result.is_ok(), true);
        let (_remain, _bytes_result) = result.unwrap();
        println!("{:?}", _remain);
        // assert_eq!(_bytes_result.to_bytes(), vec![16, 3, 2])
    }

    #[test]
    fn test_parse_integer_literal() {
        let result = parse_integer_literal(CompleteStr("23"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, Token::I32Literal { value: 23 });
    }

    #[test]
    fn test_literal_const_i32() {
        let result = parse_unary_assignment(CompleteStr("%0 = crt.literal.const.i32! 13 : i32\n"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        // TODO maybe shrink this memory
        assert_eq!(_bytes_result, vec![6, 0, 13, 0, 0, 0])
    }

    #[test]
    fn test_instruction_i32_literal() {
        // w. \n
        let result = parse_instruction(CompleteStr("%0 = crt.literal.const.i32! 13 : i32\n"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![6, 0, 13, 0, 0, 0])
    }

    #[test]
    fn test_instruction_f32_literal() {
        // w. \n
        let result = parse_instruction(CompleteStr("%0 = crt.literal.const.f32! 13.0 : f32\n"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![7, 0, 0, 0, 80, 65])
    }

    #[test]
    fn test_instruction_tensor_literal() {
        // w. \n
        let result = parse_instruction(CompleteStr(
            "%0 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>: f32\n",
        ));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(
            _bytes_result,
            vec![
                12, 0, 32, 0, 6, 0, 0, 0, 0, 0, 0, 0, 205, 204, 140, 63, 205, 204, 12, 64, 51, 51,
                83, 64, 205, 204, 140, 64, 0, 0, 176, 64, 51, 51, 211, 64, 24, 0, 2, 0, 0, 0, 0, 0,
                0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0
            ]
        )
    }

    #[test]
    fn test_instruction_assignment_matmul() {
        let result = parse_instruction(CompleteStr("%0 = crt.matmul.f32! %7, %9 : f32\n"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![13, 0, 7, 9]);

        // %86 = crt.add %82, %85 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
        let result = parse_mock_instruction(CompleteStr("%86 = crt.matmul %82, %85 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![13, 86, 82, 85]);
    }

    #[test]
    fn test_instruction_constant() {
        // %8 = crt.constant : () -> tensor<10xf32>
        let result =
            parse_mock_instruction(CompleteStr("%8 = crt.constant : () -> tensor<10xf32>"));
        let ref_result = parse_instruction(CompleteStr(
            "%8 = crt.helper.svalue.tensor! zeros<[10]>: f32\n",
        ));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        let _ref_bytes_result = ref_result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, _ref_bytes_result);
    }

    #[test]
    fn test_instruction_tensor_literal_with_zeros_helper() {
        let result = parse_instruction(CompleteStr(
            "%0 = crt.helper.svalue.tensor! zeros<[2 3]>: f32\n",
        ));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
    }

    #[test]
    fn test_instruction_tensor_literal_with_ones_helper() {
        let result = parse_instruction(CompleteStr(
            "%0 = crt.helper.svalue.tensor! ones<[2 3]>: f32\n",
        ));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
    }

    #[test]
    fn test_instruction_tensor_literal_with_uniform_helper() {
        let result = parse_instruction(CompleteStr(
            "%0 = crt.helper.rng.tensor! uniform<[2 3]>: f32\n",
        ));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
    }

    #[test]
    fn test_instruction_tensor_literal_with_normal_helper() {
        let result = parse_instruction(CompleteStr(
            "%0 = crt.helper.rng.tensor! normal<[2 3]>: f32\n",
        ));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
    }

    #[test]
    fn test_parse_reshape() {
        let result = parse_unary_assignment_with_shape_argument(CompleteStr(
            "%0 = crt.reshape! %1, [2 3]\n",
        ));
        assert_eq!(result.is_ok(), true);
        assert_eq!(result.unwrap().0.is_empty(), true);
    }

    #[test]
    fn test_parse_transpose() {
        let result = parse_unary_assignment_with_shape_argument(CompleteStr(
            "%0 = crt.transpose! %1, [2 3]\n",
        ));
        assert_eq!(result.is_ok(), true);
        let (_remain, _inst) = result.unwrap();
        assert_eq!(_remain.is_empty(), true);
        _inst.to_bytes();
    }

    #[test]
    fn test_parse_reshape_inst() {
        let result = parse_instruction(CompleteStr("%0 = crt.reshape! %1, [2 3]\n"));
        assert_eq!(result.is_ok(), true);
        assert_eq!(result.unwrap().0.is_empty(), true);
    }

    #[test]
    fn test_parse_transpose_inst() {
        let result = parse_instruction(CompleteStr("%0 = crt.transpose! %1, [2 3]\n"));
        assert_eq!(result.is_ok(), true);
        assert_eq!(result.unwrap().0.is_empty(), true);
    }

    #[test]
    fn test_parse_comment_inst() {
        let result = parse_instruction(CompleteStr("//rttransp% ose[123\n"));
        let (_remain, _parsed) = result.unwrap();
        assert_eq!(_remain.is_empty(), true);
    }
}
