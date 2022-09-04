extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

// TODO make a prelude for base to simplify the import stmts
use crate::base::constants::*;
use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_view::*;
use crate::device_context::*;
use crate::instance::*;
use crate::instruction::*;

// TODO make OpCode and Functor as Trait to ensure pluggability.
pub(crate) struct Functor {
    opcode: Option<OpCode>,
    pub kernel: Option<Kernel>,
}

impl Default for Functor {
    fn default() -> Functor {
        Functor {
            opcode: Default::default(),
            kernel: Default::default(),
        }
    }
}

impl Functor {
    pub fn new() -> Functor {
        return Self {
            opcode: Default::default(),
            kernel: Default::default(),
        };
    }

    // partially bind function to the Functor, that will be applied to Self
    pub fn bind(&mut self, kernel: Kernel) -> () {
        self.kernel = Some(kernel);
    }

    pub fn wrap_kernel_specialise_attr(
        &self,
        opcode: OpCode,
    ) -> (Vec<pso::SpecializationConstant>, Vec<u8>) {
        // specialise op by opcode
        let spec_const = opcode.to_specialise_bits();
        let spec_bytes = spec_const.to_le_bytes();
        // println!("{:?}", spec_bytes);

        let spec_const_struct = pso::SpecializationConstant {
            id: 0,
            // pub struct VkSpecializationInfo
            // id --> constant id
            // start --> offset
            // end - start --> size
            range: std::ops::Range { start: 0, end: 4 },
        };

        let spec_const = vec![spec_const_struct];
        let spec_data = spec_bytes.to_vec();
        (spec_const, spec_data)
    }

    pub fn apply<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        device_context: &mut DeviceContext,
        device_instance_ref: &DeviceInstance,
        lhs_buffer_functor: DataView<concrete_backend::Backend, T>,
        rhs_buffer_functor: DataView<concrete_backend::Backend, T>,
        opcode: OpCode,
    ) -> DataView<concrete_backend::Backend, T> {
        /*
        let init_literal: Vec<T> = match T {
            f32 => vec![T.default(); lhs_buffer_functor.data_size],
            i32 => vec![i32.default(); lhs_buffer_functor.data_size],
            _ => panic!("invalid type to apply functor"),
        };
        */

        /* calculate shape of output
        For arithmetic op, shape of output should keep same as lhs operand;
        For matmul, check value of last dimension in lhs equal to first of rhs.
          And shape of output should calculated from lhs and rhs operand
                lhs: k x m x n
                rhs: n x l
                res: k x m x l
        */
        let mut res_shape = Vec::new();
        let mut res_dsize = 1;
        let lhs_shape = Cow::from(&lhs_buffer_functor.shape[..]);
        let rhs_shape = Cow::from(&rhs_buffer_functor.shape[..]);

        let lhs_shape_size = lhs_shape.len();

        let shape_transfer_to_specialization = |id: usize| {
            let dim_const_struct = pso::SpecializationConstant {
                id: (id + 1) as u32,
                range: std::ops::Range {
                    start: (id * 4 + 4) as u16,
                    end: (id * 4 + 7) as u16,
                },
            };
            dim_const_struct
        };

        let mut opcode_constant = self.wrap_kernel_specialise_attr(opcode).0;
        let mut opcode_data = self.wrap_kernel_specialise_attr(opcode).1;

        for (id, value) in lhs_shape.iter().enumerate() {
            // println!("dim id: {}; dim value: {}", id, value);
            let spec_shape = shape_transfer_to_specialization(id);
            opcode_constant.push(spec_shape);

            let value_bytes = (*value as u32).to_le_bytes();
            for x in value_bytes {
                opcode_data.push(x);
            }
        }

        for (id, value) in rhs_shape.iter().enumerate() {
            let spec_shape = shape_transfer_to_specialization(id + lhs_shape_size);
            opcode_constant.push(spec_shape);

            let value_bytes = (*value as u32).to_le_bytes();
            for x in value_bytes {
                opcode_data.push(x);
            }
        }

        let spec_constant: Cow<[pso::SpecializationConstant]> = Cow::Owned(opcode_constant);
        let spec_data: Cow<[u8]> = Cow::Owned(opcode_data);

        match opcode {
            OpCode::ADDI32
            | OpCode::SUBI32
            | OpCode::MULI32
            | OpCode::FLOORDIVI32
            | OpCode::ADDF32
            | OpCode::SUBF32
            | OpCode::MULF32
            | OpCode::DIVF32 => {
                res_shape = lhs_shape.to_vec();
                res_dsize = lhs_buffer_functor.data_size;
            }
            OpCode::MATMULF32 => {
                // assert_eq!(lhs_buffer_functor.shape.len(), 2);
                assert_eq!(rhs_shape.len(), 2);

                assert_eq!(lhs_shape[lhs_shape_size - 1], rhs_shape[0]);

                res_shape = (lhs_shape[..lhs_shape_size - 1]).to_vec();
                res_shape.push(rhs_shape[1]);
                // res_shape = lhs_shape.to_vec();
                // res_shape[0] = rhs_shape[0];

                res_dsize =
                    lhs_buffer_functor.data_size / lhs_shape[lhs_shape_size - 1] * rhs_shape[1];
                // res_dsize = lhs_buffer_functor.data_size / lhs_shape[0] * rhs_shape[0];
            }
            _ => {
                res_shape = lhs_shape.to_vec();
                res_dsize = lhs_buffer_functor.data_size;
            }
        }
        // println!("res shape: {:?}", res_shape);

        let mut res_buffer_functor = DataView::<concrete_backend::Backend, T>::new(
            &device_context.device,
            &device_instance_ref.memory_property().memory_types,
            vec![Default::default(); res_dsize as usize],
            ElementType::F32,
            res_shape,
        );

        // TODO refactor into BufferView
        let _BINDING_ID = 0;
        // step 6 create desc_set_layout
        let descriptor_set_layout_lhs = unsafe {
            device_context.device.create_descriptor_set_layout(
                iter::once(pso::DescriptorSetLayoutBinding {
                    binding: _BINDING_ID,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                }),
                iter::empty(),
            )
        }
        .expect("Can't create descriptor set layout");
        // alloc desc sets
        let desc_set_lhs = unsafe {
            let mut desc_set = device_context
                .descriptor_pool
                .allocate_one(&descriptor_set_layout_lhs)
                .unwrap();
            device_context
                .device
                .write_descriptor_set(pso::DescriptorSetWrite {
                    set: &mut desc_set,
                    binding: _BINDING_ID,
                    array_offset: 0,
                    descriptors: iter::once(pso::Descriptor::Buffer(
                        &lhs_buffer_functor.device_buffer.buffer,
                        buffer::SubRange::WHOLE,
                    )),
                });
            desc_set
        };

        // experiments
        let descriptor_set_layout_rhs = unsafe {
            device_context.device.create_descriptor_set_layout(
                iter::once(pso::DescriptorSetLayoutBinding {
                    binding: _BINDING_ID,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                }),
                iter::empty(),
            )
        }
        .expect("Can't create descriptor set layout");
        // alloc desc sets
        let desc_set_rhs = unsafe {
            let mut desc_set = device_context
                .descriptor_pool
                .allocate_one(&descriptor_set_layout_rhs)
                .unwrap();
            device_context
                .device
                .write_descriptor_set(pso::DescriptorSetWrite {
                    set: &mut desc_set,
                    binding: _BINDING_ID,
                    array_offset: 0,
                    descriptors: iter::once(pso::Descriptor::Buffer(
                        &rhs_buffer_functor.device_buffer.buffer,
                        buffer::SubRange::WHOLE,
                    )),
                });
            desc_set
        };

        // experiments
        let descriptor_set_layout_outs = unsafe {
            device_context.device.create_descriptor_set_layout(
                iter::once(pso::DescriptorSetLayoutBinding {
                    binding: _BINDING_ID,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                }),
                iter::empty(),
            )
        }
        .expect("Can't create descriptor set layout");
        // alloc desc sets
        let desc_set_outs = unsafe {
            let mut desc_set = device_context
                .descriptor_pool
                .allocate_one(&descriptor_set_layout_outs)
                .unwrap();
            device_context
                .device
                .write_descriptor_set(pso::DescriptorSetWrite {
                    set: &mut desc_set,
                    binding: _BINDING_ID,
                    array_offset: 0,
                    descriptors: iter::once(pso::Descriptor::Buffer(
                        &res_buffer_functor.device_buffer.buffer,
                        buffer::SubRange::WHOLE,
                    )),
                });
            desc_set
        };

        // step 7 create pipeline_layout
        let pipeline_layout = unsafe {
            device_context.device.create_pipeline_layout(
                [
                    &descriptor_set_layout_lhs,
                    &descriptor_set_layout_rhs,
                    &descriptor_set_layout_outs,
                ]
                .into_iter(),
                iter::empty(),
            )
            //unsafe { device.create_pipeline_layout(iter::once(&descriptor_set_layout_lhs), iter::empty())
        }
        .expect("Can't create pipeline layout");

        // step 8 create pipeline
        // specialization: pso::Specialization::default()
        let entry_point = pso::EntryPoint {
            entry: "main",
            module: self.kernel.as_ref().unwrap(),
            specialization: pso::Specialization {
                // constants: self.wrap_kernel_specialise_attr(opcode).0,
                constants: spec_constant,
                // data: self.wrap_kernel_specialise_attr(opcode).1,
                data: spec_data,
            },
        };
        let pipeline = unsafe {
            device_context.device.create_compute_pipeline(
                &pso::ComputePipelineDesc::new(entry_point, &pipeline_layout),
                None,
            )
        }
        .expect("Error creating compute pipeline!");

        let mut command_pool = unsafe {
            device_context.device.create_command_pool(
                device_instance_ref.queue_family().id(),
                pool::CommandPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create command pool");
        let mut fence = device_context.device.create_fence(false).unwrap();

        unsafe {
            let mut command_buffer = command_pool.allocate_one(command::Level::Primary);
            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            // move ins data
            command_buffer.copy_buffer(
                &lhs_buffer_functor.host_buffer.buffer,
                &lhs_buffer_functor.device_buffer.buffer,
                iter::once(command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: F32STRIDE as u64 * lhs_buffer_functor.data_size as u64,
                }),
            );
            command_buffer.copy_buffer(
                &rhs_buffer_functor.host_buffer.buffer,
                &rhs_buffer_functor.device_buffer.buffer,
                iter::once(command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: F32STRIDE as u64 * rhs_buffer_functor.data_size as u64,
                }),
            );

            // ensure ins are all copied
            command_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::COMPUTE_SHADER,
                memory::Dependencies::empty(),
                iter::once(memory::Barrier::Buffer {
                    states: buffer::Access::TRANSFER_WRITE
                        ..buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE,
                    families: None,
                    target: &lhs_buffer_functor.device_buffer.buffer,
                    range: buffer::SubRange::WHOLE,
                }),
            );
            command_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::COMPUTE_SHADER,
                memory::Dependencies::empty(),
                iter::once(memory::Barrier::Buffer {
                    states: buffer::Access::TRANSFER_WRITE
                        ..buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE,
                    families: None,
                    target: &rhs_buffer_functor.device_buffer.buffer,
                    range: buffer::SubRange::WHOLE,
                }),
            );

            command_buffer.bind_compute_pipeline(&pipeline);

            command_buffer.bind_compute_descriptor_sets(
                &pipeline_layout,
                0,
                [&desc_set_lhs, &desc_set_rhs, &desc_set_outs].into_iter(),
                iter::empty(),
            );
            //[&desc_set_lhs, &desc_set_rhs].into_iter(),
            //iter::once(&desc_set_lhs),

            command_buffer.dispatch([3, 1, 1]);

            command_buffer.pipeline_barrier(
                pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::TRANSFER,
                memory::Dependencies::empty(),
                iter::once(memory::Barrier::Buffer {
                    states: buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE
                        ..buffer::Access::TRANSFER_READ,
                    families: None,
                    target: &res_buffer_functor.device_buffer.buffer,
                    range: buffer::SubRange::WHOLE,
                }),
            );

            // move outs data
            command_buffer.copy_buffer(
                &res_buffer_functor.device_buffer.buffer,
                &res_buffer_functor.host_buffer.buffer,
                iter::once(command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: F32STRIDE as u64 * res_buffer_functor.data_size as u64,
                }),
            );

            command_buffer.finish();

            let mut queue_group = device_context.queue_groups.first_mut().unwrap();
            queue_group.queues[0].submit(
                iter::once(&command_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut fence),
            );

            device_context.device.wait_for_fence(&fence, !0).unwrap();
            command_pool.free(iter::once(command_buffer));
        }

        unsafe {
            // device_context.device.destroy_shader_module(shader);
            device_context.device.destroy_command_pool(command_pool);
            device_context.device.destroy_fence(fence);

            device_context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout_lhs);
            device_context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout_rhs);
            device_context
                .device
                .destroy_buffer(lhs_buffer_functor.device_buffer.buffer);
            device_context
                .device
                .destroy_buffer(lhs_buffer_functor.host_buffer.buffer);
            device_context
                .device
                .destroy_buffer(rhs_buffer_functor.device_buffer.buffer);
            device_context
                .device
                .destroy_buffer(rhs_buffer_functor.host_buffer.buffer);
            device_context
                .device
                .free_memory(lhs_buffer_functor.device_buffer.memory);
            device_context
                .device
                .free_memory(lhs_buffer_functor.host_buffer.memory);
            device_context
                .device
                .free_memory(rhs_buffer_functor.device_buffer.memory);
            device_context
                .device
                .free_memory(rhs_buffer_functor.host_buffer.memory);
            device_context
                .device
                .destroy_pipeline_layout(pipeline_layout);
            device_context.device.destroy_compute_pipeline(pipeline);
        }

        res_buffer_functor
    }
}
