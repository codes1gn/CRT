use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use raptors::prelude::*;

use crate::base::constants::*;
use crate::buffer_types::*;
use crate::vkgpu_executor::*;

use crate::base::*;

impl<T> TensorLike for TensorView<T> {}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorView<T> {
    pub data: Vec<T>,
    pub dtype: ElementType,
    pub shape: Vec<usize>,
}

impl<T> TensorView<T> {
    pub fn new(data: Vec<T>, dtype: ElementType, shape: Vec<usize>) -> Self {
        Self {
            data: data,
            dtype: dtype,
            shape: shape,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActTensorTypes {
    F32Tensor { data: TensorView<f32> },
    I32Tensor { data: TensorView<i32> },
    MockTensor { data: MockTensor },
}

impl TensorLike for ActTensorTypes {}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn dummy_test() {
        assert_eq!(0, 0);
    }
}
