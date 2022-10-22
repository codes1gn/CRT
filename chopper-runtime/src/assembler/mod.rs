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
use parse_module::*;

// interface for bytecode parsing
named!(pub parse_bytecode<CompleteStr, Program>,
    do_parse!(
        program: parse_program >> (
            program
        )
    )
);

named!(pub parse_program<CompleteStr, Program>,
    do_parse!(
        modules: many1!(parse_module) >> (
            Program {
                mods: modules
            }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;
}
