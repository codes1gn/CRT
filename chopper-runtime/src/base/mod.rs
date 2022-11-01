pub mod constants;
pub mod errors;
pub mod kernel;

use nom::types::CompleteStr;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ElementType {
    I32,
    I64,
    F32,
}

impl From<CompleteStr<'_>> for ElementType {
    fn from(s: CompleteStr<'_>) -> Self {
        match s {
            CompleteStr("i32") => ElementType::I32,
            CompleteStr("i64") => ElementType::I64,
            CompleteStr("f32") => ElementType::F32,
            _ => panic!("not recognise this element type"),
        }
    }
}

impl From<u8> for ElementType {
    fn from(s: u8) -> Self {
        match s {
            0 => ElementType::I32,
            1 => ElementType::I64,
            2 => ElementType::F32,
            _ => panic!("not recognise this element type"),
        }
    }
}

impl From<&ElementType> for u8 {
    fn from(s: &ElementType) -> Self {
        match s {
            ElementType::I32 => 0,
            ElementType::I64 => 1,
            ElementType::F32 => 2,
            _ => panic!("not recognise this element type"),
        }
    }
}

pub trait SupportedType {
    fn get_type_code(&self) -> ElementType;
}

impl SupportedType for i32 {
    fn get_type_code(&self) -> ElementType {
        return ElementType::I32;
    }
}

impl SupportedType for i64 {
    fn get_type_code(&self) -> ElementType {
        return ElementType::I64;
    }
}

impl SupportedType for f32 {
    fn get_type_code(&self) -> ElementType {
        return ElementType::F32;
    }
}
