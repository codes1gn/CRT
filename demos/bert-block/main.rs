extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

fn bert_block() {
    // use shared reshape result
    let bytecode2 = "
        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        %1 = crt.reshape! %0, [4096 1024]

        // Q branch
        phantom.block <dev:#0> {
            %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %3 = crt.matmul.f32! %1, %2 : f32
            %5 = crt.add.f32! %3, %4: f32
            %6 = crt.reshape! %5, [32 128 16 64]
            %101 = crt.transpose! %6, [32 16 128 64]
        }

        // V branch
        phantom.block <dev:#1> {
            %12 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %14 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %13 = crt.matmul.f32! %1, %12 : f32
            %15 = crt.add.f32! %13, %14: f32
            %16 = crt.reshape! %15, [32 128 16 64]
            %103 = crt.transpose! %16, [32 16 128 64]
        }

        // K branch
        phantom.block <dev:#2> {
            %7 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %9 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %8 = crt.matmul.f32! %1, %7 : f32
            %10 = crt.add.f32! %8, %9: f32
            %11 = crt.reshape! %10, [32 128 16 64]
            %102 = crt.transpose! %11, [32 16 128 64]
        }

        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        %1 = crt.reshape! %0, [4096 1024]

        // Q branch
        phantom.block <dev:#0> {
            %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %3 = crt.matmul.f32! %1, %2 : f32
            %5 = crt.add.f32! %3, %4: f32
            %6 = crt.reshape! %5, [32 128 16 64]
            %101 = crt.transpose! %6, [32 16 128 64]
        }

        // V branch
        phantom.block <dev:#1> {
            %12 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %14 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %13 = crt.matmul.f32! %1, %12 : f32
            %15 = crt.add.f32! %13, %14: f32
            %16 = crt.reshape! %15, [32 128 16 64]
            %103 = crt.transpose! %16, [32 16 128 64]
        }

        // K branch
        phantom.block <dev:#2> {
            %7 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %9 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %8 = crt.matmul.f32! %1, %7 : f32
            %10 = crt.add.f32! %8, %9: f32
            %11 = crt.reshape! %10, [32 128 16 64]
            %102 = crt.transpose! %11, [32 16 128 64]
        }

        return %103
    ";

    // avoid shared use of reshape result
    let bytecode1 = "
        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        // phantom.block <dev:#0> {
        //     %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        // }
        //
        // %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        // %1 = crt.reshape! %0, [4096 1024]
        //
        // phantom.block <dev:#3> {
        //     %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        //     %1 = crt.reshape! %0, [4096 1024]
        // }
        //

        // Q branch
        phantom.block <dev:#0> {
            %2 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %4 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %113 = crt.reshape! %0, [4096 1024]
            %3 = crt.matmul.f32! %113, %2 : f32
            %5 = crt.add.f32! %3, %4: f32
            %6 = crt.reshape! %5, [32 128 16 64]
            %101 = crt.transpose! %6, [32 16 128 64]
        }

        // V branch
        phantom.block <dev:#1> {
            %12 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %14 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %117 = crt.reshape! %0, [4096 1024]
            %13 = crt.matmul.f32! %117, %12 : f32
            %15 = crt.add.f32! %13, %14: f32
            %16 = crt.reshape! %15, [32 128 16 64]
            %103 = crt.transpose! %16, [32 16 128 64]
        }

        // K branch
        phantom.block <dev:#2> {
            %7 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %9 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %127 = crt.reshape! %0, [4096 1024]
            %8 = crt.matmul.f32! %127, %7 : f32
            %10 = crt.add.f32! %8, %9: f32
            %11 = crt.reshape! %10, [32 128 16 64]
            %102 = crt.transpose! %11, [32 16 128 64]
        }

        return %103
    ";
    let mut ipt = Interpreter::new();
    ipt.init(3);

    // split reshape
    ipt.run_bytecode_lazily(bytecode1);

    // split after reshape
    // ipt.run_bytecode_lazily(bytecode2);

    // ipt.vm.dump_tensor_f32(10);
    // ipt.vm.dump_tensor_f32(11);
    // ipt.vm.dump_tensor_f32(102);
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::try_init().unwrap();
    bert_block();
}
