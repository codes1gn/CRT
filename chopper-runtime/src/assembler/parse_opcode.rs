// external crates
use nom::types::CompleteStr;
use nom::*;

// mods from local crate
use crate::instruction::CRTOpCode;

// use assembler_base::*;
use crate::assembler::assembler_base::Token;

// opcode
named!(pub parse_opcode<CompleteStr, Token>,
    do_parse!(
        // use ! tag to specify the bytecode opcode for simplicity
        opcode: take_until_and_consume1!("!")
        >> (Token::BytecodeOpCode { code: CRTOpCode::from(opcode) })
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_to_opcode() {
        let opcode = CRTOpCode::from(CompleteStr("load"));
        assert_eq!(opcode, CRTOpCode::LOAD);
        let opcode = CRTOpCode::from(CompleteStr("crt.add.i32"));
        assert_eq!(opcode, CRTOpCode::ADDI32);
        let opcode = CRTOpCode::from(CompleteStr("crt.sub.i32"));
        assert_eq!(opcode, CRTOpCode::SUBI32);
        let opcode = CRTOpCode::from(CompleteStr("crt.mul.i32"));
        assert_eq!(opcode, CRTOpCode::MULI32);
        let opcode = CRTOpCode::from(CompleteStr("crt.matmul.f32"));
        assert_eq!(opcode, CRTOpCode::MATMULF32);
    }

    #[test]
    fn test_parse_halt() {
        // test halt
        let result = parse_opcode(CompleteStr("halt!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::HALT
            }
        );
    }

    #[test]
    fn test_load_op() {
        let result = parse_opcode(CompleteStr("lload!"));
        assert_eq!(result.is_ok(), true);
        let (rest, token) = result.unwrap();
        assert_eq!(
            token,
            Token::BytecodeOpCode {
                code: CRTOpCode::ILLEGAL
            }
        );
        assert_eq!(rest, CompleteStr(""));
        let result = parse_opcode(CompleteStr("lload!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::ILLEGAL
            }
        );
        let result = parse_opcode(CompleteStr("l oad!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::ILLEGAL
            }
        );
        // case sensitive
        let result = parse_opcode(CompleteStr("Load!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::ILLEGAL
            }
        );
        let result = parse_opcode(CompleteStr("LoAd!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::ILLEGAL
            }
        );
        // test load
        let result = parse_opcode(CompleteStr("load!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::LOAD
            }
        );
    }

    #[test]
    fn test_parse_binary_code() {
        // test add
        let result = parse_opcode(CompleteStr("crt.add.i32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::ADDI32
            }
        );
        // test sub
        let result = parse_opcode(CompleteStr("crt.sub.i32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::SUBI32
            }
        );
        // test mul
        let result = parse_opcode(CompleteStr("crt.mul.i32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::MULI32
            }
        );
        // test floordiv i32
        let result = parse_opcode(CompleteStr("crt.floordiv.i32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::FLOORDIVI32
            }
        );
        // test add
        let result = parse_opcode(CompleteStr("crt.add.f32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::ADDF32
            }
        );
        // test sub
        let result = parse_opcode(CompleteStr("crt.sub.f32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::SUBF32
            }
        );
        // test mul
        let result = parse_opcode(CompleteStr("crt.mul.f32!"));
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::MULF32
            }
        );
        // test div f32
        let result = parse_opcode(CompleteStr("crt.div.f32!"));
        println!("{:?}", result);
        assert_eq!(
            result.unwrap().1,
            Token::BytecodeOpCode {
                code: CRTOpCode::DIVF32
            }
        );
    }

    // test parse crt literal const op only
    #[test]
    fn test_literal_const_i32_op() {
        let result = parse_opcode(CompleteStr("crt.literal.const.i32!"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let (rest, token) = result.unwrap();
        assert_eq!(
            token,
            Token::BytecodeOpCode {
                code: CRTOpCode::CONSTI32
            }
        );
        assert_eq!(rest, CompleteStr(""));
    }
    #[test]
    fn test_literal_const_f32_op() {
        let result = parse_opcode(CompleteStr("crt.literal.const.f32!"));
        assert_eq!(result.is_ok(), true);
        let (rest, token) = result.unwrap();
        assert_eq!(
            token,
            Token::BytecodeOpCode {
                code: CRTOpCode::CONSTF32
            }
        );
        assert_eq!(rest, CompleteStr(""));
    }
}
