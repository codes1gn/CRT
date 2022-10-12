use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use raptors::prelude::*;

#[cfg(any(feature = "mock", feature = "blas"))]
use rublas::prelude::*;

use crate::base::constants::*;
use crate::buffer_types::*;
use crate::vkgpu_executor::*;

use crate::base::*;

#[derive(Debug, Clone, PartialEq)]
pub enum ActTensorTypes {
    F32Tensor { data: TensorView<f32> },
    I32Tensor { data: TensorView<i32> },
    MockTensor { data: MockTensor },
}

impl TensorLike for ActTensorTypes {}

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

impl<T> TensorLike for TensorView<T> {}

#[cfg(any(feature = "mock", feature = "blas"))]
impl From<TensorView<f32>> for BlasTensor {
    fn from(item: TensorView<f32>) -> Self {
        BlasTensor::from_vec_shape(item.data, item.shape)
    }
}

// TODO verify this impl of conversion, that will not omit excessive time/memory cost
#[cfg(any(feature = "mock", feature = "blas"))]
impl From<&TensorView<f32>> for BlasTensor {
    fn from(item: &TensorView<f32>) -> Self {
        BlasTensor::from_vec_shape((*item.data).to_vec(), (*item.shape).to_vec())
    }
}

#[cfg(any(feature = "mock", feature = "blas"))]
impl From<TensorView<i32>> for BlasTensor {
    fn from(item: TensorView<i32>) -> Self {
        BlasTensor::from_vec_shape_i32(item.data, item.shape)
    }
}

#[cfg(any(feature = "mock", feature = "blas"))]
impl From<&TensorView<i32>> for BlasTensor {
    fn from(item: &TensorView<i32>) -> Self {
        BlasTensor::from_vec_shape_i32((*item.data).to_vec(), (*item.shape).to_vec())
    }
}

#[cfg(any(feature = "mock", feature = "blas"))]
impl From<BlasTensor> for TensorView<f32> {
    fn from(item: BlasTensor) -> Self {
        match item.data {
            TensorKind::FloatVector(data) => {
                TensorView::<f32>::new(data.into_raw_vec(), ElementType::F32, item.shape)
            }
            TensorKind::FloatMatrix(data) => {
                TensorView::<f32>::new(data.into_raw_vec(), ElementType::F32, item.shape)
            }
            _ => panic!("not supported dtype"),
        }
    }
}

#[cfg(any(feature = "mock", feature = "blas"))]
impl From<BlasTensor> for TensorView<i32> {
    fn from(item: BlasTensor) -> Self {
        match item.data {
            TensorKind::Int32Vector(data) => {
                TensorView::<i32>::new(data.into_raw_vec(), ElementType::I32, item.shape)
            }
            TensorKind::Int32Matrix(data) => {
                TensorView::<i32>::new(data.into_raw_vec(), ElementType::I32, item.shape)
            }
            _ => panic!("not supported dtype"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_test() {
        assert_eq!(0, 0);
    }
}
