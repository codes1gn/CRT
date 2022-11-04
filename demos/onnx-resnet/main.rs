extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;
// module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
//   func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> attributes {input_names = ["data"], output_names = ["resnetv15_dense0_fwd"]} {
//
//   }
//   "onnx.EntryPoint"() {func = @main_graph} : () -> ()
// }
fn resnet() {
    // avoid shared use of reshape result
    let bytecode = r#"
    %0 = crt.constant : () -> tensor<1000x512xf32>
    %1 = crt.constant : () -> tensor<1000xf32>
    %2 = crt.constant : () -> tensor<64x3x7x7xf32>
    %3 = crt.constant : () -> tensor<64xf32>
    %4 = crt.constant : () -> tensor<64x64x3x3xf32>
    %5 = crt.constant : () -> tensor<64xf32>
    %6 = crt.constant : () -> tensor<64x64x3x3xf32>
    %7 = crt.constant : () -> tensor<64xf32>
    %8 = crt.constant : () -> tensor<64x64x3x3xf32>
    %9 = crt.constant : () -> tensor<64xf32>
    %10 = crt.constant : () -> tensor<64x64x3x3xf32>
    %11 = crt.constant : () -> tensor<64xf32>
    %12 = crt.constant : () -> tensor<128x64x1x1xf32>
    %13 = crt.constant : () -> tensor<128xf32>
    %14 = crt.constant : () -> tensor<128x64x3x3xf32>
    %15 = crt.constant : () -> tensor<128xf32>
    %16 = crt.constant : () -> tensor<128x128x3x3xf32>
    %17 = crt.constant : () -> tensor<128xf32>
    %18 = crt.constant : () -> tensor<128x128x3x3xf32>
    %19 = crt.constant : () -> tensor<128xf32>
    %20 = crt.constant : () -> tensor<128x128x3x3xf32>
    %21 = crt.constant : () -> tensor<128xf32>
    %22 = crt.constant : () -> tensor<256x128x1x1xf32>
    %23 = crt.constant : () -> tensor<256xf32>
    %24 = crt.constant : () -> tensor<256x128x3x3xf32>
    %25 = crt.constant : () -> tensor<256xf32>
    %26 = crt.constant : () -> tensor<256x256x3x3xf32>
    %27 = crt.constant : () -> tensor<256xf32>
    %28 = crt.constant : () -> tensor<256x256x3x3xf32>
    %29 = crt.constant : () -> tensor<256xf32>
    %30 = crt.constant : () -> tensor<256x256x3x3xf32>
    %31 = crt.constant : () -> tensor<256xf32>
    %32 = crt.constant : () -> tensor<512x256x1x1xf32>
    %33 = crt.constant : () -> tensor<512xf32>
    %34 = crt.constant : () -> tensor<512x256x3x3xf32>
    %35 = crt.constant : () -> tensor<512xf32>
    %36 = crt.constant : () -> tensor<512x512x3x3xf32>
    %37 = crt.constant : () -> tensor<512xf32>
    %38 = crt.constant : () -> tensor<512x512x3x3xf32>
    %39 = crt.constant : () -> tensor<512xf32>
    %40 = crt.constant : () -> tensor<512x512x3x3xf32>
    %41 = crt.constant : () -> tensor<512xf32>
    %42 = crt.convadd %0, %2, %3 : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %43 = crt.relu %42 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %44 = crt.maxpool %43 : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    %45 = crt.convadd %44, %4, %5 : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %46 = crt.relu %45 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %47 = crt.convadd %46, %6, %7 : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %48 = crt.add %44, %47 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %49 = crt.relu %48 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %50 = crt.convadd %49, %8, %9 : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %51 = crt.relu %50 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %52 = crt.convadd %51, %10, %11 : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %53 = crt.add %49, %52 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %54 = crt.relu %53 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %55 = crt.convadd %54, %12, %13 : (tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %56 = crt.convadd %54, %14, %15 : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %57 = crt.relu %56 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %58 = crt.convadd %57, %16, %17 : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %59 = crt.add %55, %58 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %60 = crt.relu %59 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %61 = crt.convadd %60, %18, %19 : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %62 = crt.relu %61 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %63 = crt.convadd %62, %20, %21 : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %64 = crt.add %60, %63 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %65 = crt.relu %64 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %66 = crt.convadd %65, %22, %23 : (tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %67 = crt.convadd %65, %24, %25 : (tensor<1x128x28x28xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %68 = crt.relu %67 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %69 = crt.convadd %68, %26, %27 : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %70 = crt.add %66, %69 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %71 = crt.relu %70 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %72 = crt.convadd %71, %28, %29 : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %73 = crt.relu %72 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %74 = crt.convadd %73, %30, %31 : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %75 = crt.add %71, %74 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %76 = crt.relu %75 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %77 = crt.convadd %76, %32, %33 : (tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %78 = crt.convadd %76, %34, %35 : (tensor<1x256x14x14xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %79 = crt.relu %78 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %80 = crt.convadd %79, %36, %37 : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %81 = crt.add %77, %80 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %82 = crt.relu %81 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %83 = crt.convadd %82, %38, %39 : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %84 = crt.relu %83 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %85 = crt.convadd %84, %40, %41 : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %86 = crt.add %82, %85 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %87 = crt.relu %86 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %88 = crt.reducemean %87 : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
    %89 = crt.flatten %88 : (tensor<1x512x1x1xf32>) -> tensor<1x512xf32>
    %90 = crt.gemm %89, %0, %1 : (tensor<1x512xf32>, tensor<1000x512xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    return %90 : tensor<1x1000xf32>
    "#;
    let mut ipt = Interpreter::new();
    ipt.init(3);

    // split reshape
    ipt.run_bytecode_lazily(bytecode);
    ipt.vm.dump_tensor_f32(90);
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::try_init().unwrap();
    resnet();
}
