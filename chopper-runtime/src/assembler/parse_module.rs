// external crates
use nom::types::CompleteStr;
use nom::*;

use super::assembler_base::*;
use super::parse_instruction::*;

// phantom module is the runtime support in bytecode format that accepts
// optimised phantom ir
named!(pub parse_phantom_module<CompleteStr, Module>,
    do_parse!(
        opt!(multispace) >>
        tag!("phantom.block") >>
        space0 >>
        tag!("<dev:#") >>
        _dev_at: digit >>
        tag!("> {") >>
        instructions: many1!(
            parse_instruction
        ) >>
        tag!("}") >>
        opt!(multispace) >>
        (
            Module {
                dev_at: Some(_dev_at.parse::<u8>().unwrap()),
                instructions: instructions,
            }
        )
    )
);

named!(pub parse_module<CompleteStr, Module>,
    do_parse!(
        instructions: many1!(
            parse_instruction
        ) >>
        (
            Module {
                dev_at: None,
                instructions: instructions
            }
        )
    )
);

#[cfg(feature = "mock")]
named!(pub parse_mock_module<CompleteStr, Module>,
    do_parse!(
        instructions: many1!(
            parse_mock_instruction
        ) >>
        (
            Module {
                dev_at: None,
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
        // first annotate devat
        assert_eq!(
            _bytes_result,
            vec![21, 255, 6, 0, 13, 0, 0, 0, 6, 0, 13, 0, 0, 0]
        )
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

    #[test]
    fn test_parse_module() {
        let bytecode = "
            %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
            %1 = crt.reshape! %0, [4096 1024]
            %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %3 = crt.matmul.f32! %1, %2 : f32
            %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %5 = crt.add.f32! %3, %4: f32
            %6 = crt.reshape! %5, [32 128 16 64]
            %101 = crt.transpose! %6, [32 16 128 64]
        ";
        let full_result = parse_module(CompleteStr(bytecode));
        assert_eq!(full_result.is_ok(), true);
        let (_remain, _parsed) = full_result.unwrap();
        assert_eq!(_remain.is_empty(), true);
    }

    #[test]
    fn test_parse_phantom_module() {
        let bytecode = "
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
        ";
        let full_result = parse_phantom_module(CompleteStr(bytecode));
        // assert_eq!(full_result.is_ok(), true);
        let (_remain, _parsed) = full_result.unwrap();
        println!("{:?}", _remain);
        println!("{:?}", _parsed);
        assert_eq!(_remain.is_empty(), true);
        assert_eq!(_parsed.dev(), Some(1));
    }
}
