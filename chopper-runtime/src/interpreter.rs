extern crate float_eq;
use std::{thread, time};

use float_eq::{assert_float_eq, float_eq};

// use tracing related mods
// replace log with tracing
//
// use log::{debug, info};

#[cfg(any(feature = "blas", feature = "phantom", feature = "mock"))]
use tracing::{debug, info};

// use opentelemetry::global;
// use tracing_subscriber::prelude::*;
// use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

use nom::types::CompleteStr;
use std;
use std::io;
use std::io::Write;
use std::num::ParseIntError;

use crate::assembler::parse_bytecode;
use crate::base::errors::*;
use crate::instance::*;
use crate::vm::VM;

#[derive(Debug)]
pub struct Interpreter {
    history: Vec<String>,
    pub vm: VM,
}

impl Interpreter {
    pub fn new() -> Interpreter {
        // let ist = DeviceInstance::new();
        Interpreter {
            history: vec![],
            vm: VM::new(),
        }
    }

    pub fn init(&mut self, executor_cnt: usize) {
        // init tracing configuration
        // #[cfg(any(feature = "mock", feature = "blas"))]
        // std::env::set_var("RUST_LOG", "info");
        // if std::env::args().any(|arg| arg == "--trace") {
        //     global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
        //     let tracer = opentelemetry_jaeger::new_pipeline()
        //         .with_service_name("raptors")
        //         .install_simple()
        //         .unwrap();

        //     let opentelemetry = tracing_opentelemetry::layer().with_tracer(tracer);
        //     tracing_subscriber::registry()
        //         .with(opentelemetry)
        //         .with(fmt::Layer::default())
        //         .try_init()
        //         .unwrap();
        // } else {
        //     tracing_subscriber::fmt::try_init().unwrap();
        //     // env_logger.init();
        // };
        info!(" == CRT IPT initialization done == ");

        // init vm environments
        self.vm.init(executor_cnt);
    }

    /// Accepts a hexadecimal string WITHOUT a leading `0x` and returns a Vec of u8
    /// Example for a LOAD command: 00 01 03 E8
    /// TODO add this attr, to ensure its deprecated
    // #[allow(dead_code)]
    fn parse_hex(&mut self, i: &str) -> Result<Vec<u8>, ParseIntError> {
        let split = i.split(' ').collect::<Vec<&str>>();
        let mut results: Vec<u8> = vec![];
        for hex_string in split {
            let byte = u8::from_str_radix(hex_string, 16);
            match byte {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        // TODO currently only allow for 4 bytes length format of instruction
        // to be extended later
        assert_eq!(results.len(), 4);
        Ok(results)
    }

    pub fn run_bytecode_eagerly(&mut self, bytecode: &str) -> Result<u8, RuntimeStatusError> {
        let parsed_program = parse_bytecode(CompleteStr(bytecode));
        // add assertion here to avoid semantic error in bytecode
        assert_eq!(parsed_program.is_ok(), true);
        let (_remain, result_program) = parsed_program.unwrap();
        assert_eq!(_remain.is_empty(), true);
        let bytecode = result_program.to_bytes();
        for byte in bytecode {
            self.vm.push_instruction(byte);
        }
        let status = self.vm.run_eagerly();
        status
    }

    pub fn run_bytecode_lazily(&mut self, bytecode: &str) -> Result<u8, RuntimeStatusError> {
        let parsed_program = parse_bytecode(CompleteStr(bytecode));
        assert_eq!(parsed_program.is_ok(), true);
        let (_remain, result_program) = parsed_program.expect("failed to parse bytecode");
        info!("remain program = {:?}", _remain);
        assert_eq!(_remain.is_empty(), true);
        let bytecode = result_program.to_bytes();
        for byte in bytecode {
            self.vm.push_instruction(byte);
        }
        let status = self.vm.run_lazily();
        status
    }

    fn consume_command(&mut self, bytecode: &str) -> Result<u8, RuntimeStatusError> {
        match bytecode {
            "exit" | "quit" | "q" => {
                info!("Chopper-Runtime Halt Now");
                // TODO make put this setting to base const, halt exit code use 1, else use 0
                Ok(7)
            }
            "history" | "h" => {
                for cmd in &self.history {
                    info!("|-- {}", cmd);
                }
                // TODO history cmd use 6 as status code
                Ok(6)
            }
            "list" | "l" => {
                info!("action: Showing instruction queue");
                for inst in self.vm.inst_buffer() {
                    info!("|-- {}", inst);
                }
                // TODO
                Ok(5)
            }
            "display" | "watch" | "wt" => {
                info!("action: Showing registers");
                let mut reg_table = vec![];
                for reg in self.vm.registers() {
                    reg_table.push(reg.clone());
                }
                info!("{:#?}", reg_table);
                // TODO
                Ok(4)
            }
            _ => self.run_bytecode_eagerly(bytecode),
        }
    }

    pub fn run(&mut self) {
        info!("~~~~~~~~~  Entering Chopper Runtime ~~~~~~~~~~");
        loop {
            let mut bytecode = String::new();
            let stdin = io::stdin();

            // show >> prompts
            print!(">> ");
            io::stdout().flush().expect("error: Failed to print");

            // blocking until inputs come
            stdin
                .read_line(&mut bytecode)
                .expect("error: Failed to read user inputs");
            let bytecode = bytecode.trim();
            // after handling this command, add it to the history list
            let step_status = self.consume_command(&bytecode);
            match step_status {
                Ok(_) => info!("::ipt::should not happen - status ok"),
                Err(e) => match e {
                    RuntimeStatusError::RT_ERROR => panic!("::ipt::RT status incorrect"),
                    // TODO use OK rather Err for program_finish
                    RuntimeStatusError::EXEC_FINISH => info!("::ipt::computation-finish"),
                },
            }
            self.history.push(bytecode.to_string());
        }
        info!("~~~~~~~~ Exiting Chopper Runtime ~~~~~~~~");
    }
}

#[cfg(not(feature = "mock"))]
#[cfg(not(feature = "phantom"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_interpreter() {
        //let ipt = Interpreter::new();
        let mut ipt = Interpreter::new();
        assert_eq!(ipt.history.len(), 0);
    }

    #[test]
    fn test_push_history() {
        let mut ipt = Interpreter::new();
        let fake_cmd = String::from("exit");
        ipt.history.push(fake_cmd.to_string());
        assert_eq!(ipt.history[0], "exit");
    }

    #[test]
    fn test_mock_bytecode_halt() {
        let mut ipt = Interpreter::new();
        let status = ipt.run_bytecode_eagerly("halt");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_i32_literal() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        // TODO make runtime check on matching const.i32 and i32 type annotation
        let status = ipt.run_bytecode_eagerly("%17 = crt.literal.const.i32! 13 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(17), vec![13]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_f32_literal() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%8 = crt.literal.const.f32! 1.3 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_f32(8), vec![1.3]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_tensor_zeros_helper() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status =
            ipt.run_bytecode_eagerly("%8 = crt.helper.svalue.tensor! zeros<[8 3]> : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_f32(8), vec![0f32; 24]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_tensor_ones_helper() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%8 = crt.helper.svalue.tensor! ones<[8 3]> : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_f32(8), vec![1f32; 24]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_tensor_uniform_helper() {
        let mut ipt = Interpreter::new();
        ipt.init(1);
        // ok
        let status = ipt.run_bytecode_eagerly("%8 = crt.helper.rng.tensor! uniform<[8 3]> : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_tensor_normal_helper() {
        let mut ipt = Interpreter::new();
        ipt.init(1);
        // ok
        let status = ipt.run_bytecode_eagerly("%8 = crt.helper.rng.tensor! normal<[8 3]> : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_add_i32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_i32(1), vec![1]);
        assert_eq!(*ipt.vm.get_raw_vec_i32(2), vec![2]);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.add.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(3), vec![3]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_sub_i32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_i32(1), vec![1]);
        assert_eq!(*ipt.vm.get_raw_vec_i32(2), vec![2]);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.sub.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(3), vec![-1]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_mul_i32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_i32(1), vec![1]);
        assert_eq!(*ipt.vm.get_raw_vec_i32(2), vec![2]);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.mul.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(3), vec![2]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_floordiv_i32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_i32(1), vec![1]);
        assert_eq!(*ipt.vm.get_raw_vec_i32(2), vec![2]);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.floordiv.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(3), vec![0]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_floordiv_i32_case2() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.i32! 7 : i32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_i32(1), vec![7]);
        assert_eq!(*ipt.vm.get_raw_vec_i32(2), vec![2]);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.floordiv.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(3), vec![3]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_add_f32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.add.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(3), vec![3.3], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_sub_f32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.sub.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(3), vec![-1.1], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_mul_f32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.mul.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(3), vec![2.42], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_div_f32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.run_bytecode_eagerly("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.run_bytecode_eagerly("%3 = crt.div.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(3), vec![0.5], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_f32_binary_add_then_sub_i32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%8 = crt.literal.const.i32! 3 : i32\n");
        let status = ipt.run_bytecode_eagerly("%7 = crt.literal.const.i32! 2 : i32\n");
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.i32! 7 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_i32(8), vec![3]);
        assert_eq!(*ipt.vm.get_raw_vec_i32(7), vec![2]);

        // add
        let status = ipt.run_bytecode_eagerly("%4 = crt.add.i32! %8, %7 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_raw_vec_i32(4), vec![5]);

        // sub
        let status = ipt.run_bytecode_eagerly("%5 = crt.sub.i32! %1, %4 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        // TODO package this assert macro into utils, hide rmax_all setting from hardcode
        assert_eq!(*ipt.vm.get_raw_vec_i32(5), vec![2]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_f32_binary_add_then_sub_f32() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly("%8 = crt.literal.const.f32! 1.3 : f32\n");
        let status = ipt.run_bytecode_eagerly("%7 = crt.literal.const.f32! 2.9 : f32\n");
        let status = ipt.run_bytecode_eagerly("%1 = crt.literal.const.f32! 7.4 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_raw_vec_f32(8), vec![1.3]);
        assert_eq!(*ipt.vm.get_raw_vec_f32(7), vec![2.9]);

        // add
        let status = ipt.run_bytecode_eagerly("%4 = crt.add.f32! %8, %7 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(4), vec![4.2], rmax_all <= 0.00001);

        // sub
        let status = ipt.run_bytecode_eagerly("%5 = crt.sub.f32! %1, %4 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        // TODO package this assert macro into utils, hide rmax_all setting from hardcode
        assert_float_eq!(*ipt.vm.get_raw_vec_f32(5), vec![3.2], rmax_all <= 0.00001);
    }

    #[test]
    fn test_mock_bytecode_tensor_add_lazy() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        let bytecode = "
            %0 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n\
            %1 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n\
            %2 = crt.helper.svalue.tensor! ones<[2 2]> : f32\n\
            %3 = crt.add.f32! %0, %1 : f32\n\
            %4 = crt.add.f32! %2, %3 : f32\n\
            return %4\n
        ";
        // ipt.run_bytecode_eagerly(bytecode);
        ipt.run_bytecode_lazily(bytecode);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(4),
            vec![3.0; 4],
            rmax_all <= 0.00001
        );
        // ok
    }

    #[test]
    fn test_mock_bytecode_tensor_add() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly(
            "%0 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>: f32\n",
        );
        let status = ipt.run_bytecode_eagerly(
            "%1 = crt.literal.const.tensor! dense<[2.2 3.3 3.3 1.1 3.3 2.2], shape=[2 3]>: f32\n",
        );
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(0),
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(0), vec![2, 3]);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(1),
            vec![2.2, 3.3, 3.3, 1.1, 3.3, 2.2],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(1), vec![2, 3]);

        // add
        let status = ipt.run_bytecode_eagerly("%4 = crt.add.f32! %0, %1 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(4),
            vec![3.3, 5.5, 6.6, 5.5, 8.8, 8.8],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_mock_bytecode_tensor_sub() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        let status = ipt.run_bytecode_eagerly(
            "%9 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>: f32\n",
        );
        let status = ipt.run_bytecode_eagerly(
            "%7 = crt.literal.const.tensor! dense<[2.2 3.3 3.3 1.1 3.3 2.2], shape=[2 3]>: f32\n",
        );
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(9),
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(9), vec![2, 3]);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(7),
            vec![2.2, 3.3, 3.3, 1.1, 3.3, 2.2],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(7), vec![2, 3]);

        // sub
        let status = ipt.run_bytecode_eagerly("%5 = crt.sub.f32! %7, %9 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(5),
            vec![1.1, 1.1, 0.0, -3.3, -2.2, -4.4],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_mock_bytecode_tensor_matmul() {
        let mut ipt = Interpreter::new();
        ipt.init(2);
        // ok
        // matmul(3x2, 2x3) => (3x3)
        let status = ipt.run_bytecode_eagerly(
            "%9 = crt.literal.const.tensor! dense<[1. 2. 3. 4. 5. 6.], shape=[2 3]>: f32\n",
        );
        let status = ipt.run_bytecode_eagerly(
            "%7 = crt.literal.const.tensor! dense<[1. 1. 1. 1. 1. 1.], shape=[3 2]>: f32\n",
        );
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(9),
            vec![1., 2., 3., 4., 5., 6.],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(9), vec![2, 3]);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(7),
            vec![1., 1., 1., 1., 1., 1.],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(7), vec![3, 2]);

        // matmul, temparilly faked with add
        let status = ipt.run_bytecode_eagerly("%5 = crt.matmul.f32! %7, %9 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(5),
            vec![5., 7., 9., 5., 7., 9., 5., 7., 9.],
            rmax_all <= 0.00001
        );

        let status = ipt.run_bytecode_eagerly(
            "%9 = crt.literal.const.tensor! dense<[1. 2. 3. 4. 5. 6.], shape=[2 3]>: f32\n",
        );
        let status = ipt.run_bytecode_eagerly(
            "%7 = crt.literal.const.tensor! dense<[1. 1. 1. 1. 1. 1.], shape=[3 2]>: f32\n",
        );
        let status = ipt.run_bytecode_eagerly("%6 = crt.matmul.f32! %9, %7 : f32\n");
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(6),
            vec![6., 6., 15., 15.],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_big_matrix_add() {
        // step 1, init device instance, also in VM instance init part
        // let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new();
        ipt.init(3);

        ipt.run_bytecode_eagerly("%0 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");
        ipt.run_bytecode_eagerly("%1 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");

        // TODO svalue<[shape], 0.7>
        // ipt.run_bytecode_eagerly("%4 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");

        ipt.run_bytecode_eagerly("%3 = crt.add.f32! %1, %0 : f32\n");
        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(3),
            vec![2.0; 34 * 82 * 3],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_reshape() {
        // step 1, init device instance, also in VM instance init part
        // let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new();
        ipt.init(1);

        ipt.run_bytecode_eagerly("%0 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");
        ipt.run_bytecode_eagerly("%1 = crt.reshape! %0, [102 82 1]\n");

        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(1),
            vec![1.0; 102 * 82 * 1],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_tensor_shape(1), vec![102, 82, 1],);
    }
}

#[cfg(feature = "mock")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() {
        // step 1, init device instance, also in VM instance init part
        // let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new();
        ipt.init(1);

        ipt.run_bytecode_eagerly("%0 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");
        ipt.run_bytecode_eagerly("%1 = crt.reshape! %0, [102 82 1]\n");

        assert_eq!(*ipt.vm.get_tensor_shape(0), vec![34, 82, 3],);

        assert_eq!(*ipt.vm.get_tensor_shape(1), vec![102, 82, 1],);

        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(1),
            vec![1.0; 102 * 82 * 1],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_reshape_inplace() {
        // step 1, init device instance, also in VM instance init part
        // let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new();
        ipt.init(1);

        ipt.run_bytecode_eagerly("%0 = crt.helper.svalue.tensor! ones<[34 82 3]> : f32\n");
        ipt.run_bytecode_eagerly("%0 = crt.reshape! %0, [102 82 1]\n");

        assert_eq!(*ipt.vm.get_tensor_shape(0), vec![102, 82, 1],);

        assert_float_eq!(
            *ipt.vm.get_raw_vec_f32(0),
            vec![1.0; 102 * 82 * 1],
            rmax_all <= 0.00001
        );
    }
}
