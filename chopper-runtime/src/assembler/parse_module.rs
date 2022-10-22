// external crates
use nom::types::CompleteStr;
use nom::*;

use super::assembler_base::*;
use super::parse_instruction::*;

named!(pub parse_module<CompleteStr, Module>,
    do_parse!(
        instructions: many1!(
            parse_instruction
        ) >> (
            Module {
                instructions: instructions
            }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_const_literal() {
        let result = parse_module(CompleteStr(
            "%0 = crt.literal.const.i32! 13 : i32\n%0 = crt.literal.const.i32! 13 : i32\n",
        ));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![6, 0, 13, 0, 0, 0, 6, 0, 13, 0, 0, 0])
    }

    #[test]
    fn test_parse_module_multiline() {
        let full_bytecode = "%0 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n%1 = crt.exp.f32! %0 : f32\n%2 = crt.exp.f32! %1 : f32\n%3 = crt.exp.f32! %2 : f32\n%4 = crt.exp.f32! %3 : f32\n";
        let seg2_bytecode = "%1 = crt.exp.f32! %0 : f32\n";
        let seg1_bytecode = "%0 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n";
        let full_result = parse_module(CompleteStr(full_bytecode));
        println!("{:#?}", full_result);
        let seg1_result = parse_module(CompleteStr(seg1_bytecode));
        println!("{:#?}", seg1_result);
        let seg2_result = parse_module(CompleteStr(seg2_bytecode));
        println!("{:#?}", seg2_result);
        assert_eq!(full_result.is_ok(), true);
        let _bytes_result = full_result.unwrap().1.to_bytes();
    }
}
