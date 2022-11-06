extern crate chopper_runtime;
extern crate float_eq;

use std::time::Instant;
use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use criterion::*;
use float_eq::{assert_float_eq, float_eq};

use chopper_runtime::prelude::*;

fn bert_block(crit: &mut Criterion) {
    // use shared reshape result
    let bytecode3 = "
        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32
        %1 = crt.reshape! %0, [4096 1024]
        %2 = crt.reshape! %0, [4096 1024]
        %3 = crt.reshape! %0, [4096 1024]

        // Q branch
        phantom.block <dev:#0> {
            %21 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %41 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %31 = crt.matmul.f32! %1, %21 : f32
            %51 = crt.add.f32! %31, %41: f32
            %61 = crt.reshape! %51, [32 128 16 64]
            %101 = crt.transpose! %61, [32 16 128 64]
        }
        
        // Q branch
        phantom.block <dev:#0> {
            %22 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %42 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %32 = crt.matmul.f32! %2, %22 : f32
            %52 = crt.add.f32! %32, %42: f32
            %62 = crt.reshape! %52, [32 128 16 64]
            %102 = crt.transpose! %62, [32 16 128 64]
        }

        // Q branch
        phantom.block <dev:#0> {
            %23 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %43 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %33 = crt.matmul.f32! %3, %23 : f32
            %53 = crt.add.f32! %33, %43: f32
            %63 = crt.reshape! %53, [32 128 16 64]
            %103 = crt.transpose! %63, [32 16 128 64]
        }


        // K branch
        phantom.block <dev:#1> {
            %121 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %141 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %131 = crt.matmul.f32! %101, %121 : f32
            %151 = crt.add.f32! %131, %141: f32
            %161 = crt.reshape! %151, [32 128 16 64]
            %201 = crt.transpose! %161, [32 16 128 64]
        }
        
        // Q branch
        phantom.block <dev:#0> {
            %122 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %142 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %132 = crt.matmul.f32! %102, %122 : f32
            %152 = crt.add.f32! %132, %142: f32
            %162 = crt.reshape! %152, [32 128 16 64]
            %202 = crt.transpose! %162, [32 16 128 64]
        }

        // Q branch
        phantom.block <dev:#0> {
            %123 = crt.helper.svalue.tensor! ones<[1024 1024]> : f32
            %143 = crt.helper.svalue.tensor! ones<[4096 1024]> : f32
            %133 = crt.matmul.f32! %103, %123 : f32
            %153 = crt.add.f32! %133, %143: f32
            %163 = crt.reshape! %153, [32 128 16 64]
            %203 = crt.transpose! %163, [32 16 128 64]
        }


        return %203
    ";
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

    // avoid shared use of reshape result
    let bytecode1 = "
        %0 = crt.helper.svalue.tensor! ones<[32 128 1024]> : f32

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

    let mut bench_group = crit.benchmark_group("bert_block-QKVbranches");
    bench_group.sample_size(10);
    bench_group.bench_with_input(
        BenchmarkId::new(format!("solution#1-dataparallel"), 0),
        &0,
        |bench, _zero| {
            bench.iter(|| {
                black_box(ipt.run_bytecode_lazily(bytecode1));
            });
        },
    );

    bench_group.bench_with_input(
        BenchmarkId::new(format!("solution#2-dataparallel"), 0),
        &0,
        |bench, _zero| {
            bench.iter(|| {
                black_box(ipt.run_bytecode_lazily(bytecode2));
            });
        },
    );

    // bench_group.bench_with_input(
    //     BenchmarkId::new(format!("solution#3-pipelineparallel"), 0),
    //     &0,
    //     |bench, _zero| {
    //         bench.iter(|| {
    //             black_box(ipt.run_bytecode_lazily(bytecode3));
    //         });
    //     },
    // );
}

criterion_group!(mnist_bench, bert_block);
criterion_main!(mnist_bench);