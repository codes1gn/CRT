extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

fn mnist() {
    // avoid shared use of reshape result
    let bytecode = r#"
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @main_graph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> attributes {input_names = ["image"], output_names = ["prediction"]} {
    %0 = crt.maxpool %arg0 : (tensor<1x1x28x28xf32>) -> tensor<1x1x14x14xf32>
    %1 = crt.constant : () -> tensor<2xi64>
    %2 = "onnx.Reshape"(%0, %1) {allowzero = 0 : si64, onnx_node_name = "Reshape_2"} : (tensor<1x1x14x14xf32>, tensor<2xi64>) -> tensor<1x196xf32>
    %3 = crt.constant : () -> tensor<128x196xf32>
    %4 = crt.constant : () -> tensor<128xf32>
    %5 = crt.gemm %2, %3, %4 : (tensor<1x196xf32>, tensor<128x196xf32>, tensor<128xf32>) -> tensor<1x128xf32>
    %6 = crt.relu %5 : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %7 = crt.constant : () -> tensor<10x128xf32>
    %8 = crt.constant : () -> tensor<10xf32>
    %9 = crt.gemm %6, %7, %8 : (tensor<1x128xf32>, tensor<10x128xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    %10 = crt.softmax %9 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %10 : tensor<1x10xf32>
  }
}
    "#;
    let mut ipt = Interpreter::new();
    ipt.init(3);

    // split reshape
    ipt.run_bytecode_lazily(bytecode);
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::try_init().unwrap();
    mnist();
}
