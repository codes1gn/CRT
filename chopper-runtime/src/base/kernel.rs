// TODO define new type instead, use alias to workaround currently
//pub(crate) struct Kernel(Vec<u32>);
extern crate backend_vulkan as concrete_backend;

// pub type Kernel = Vec<u32>;
pub type KernelByteCode = Vec<u32>;

pub type Kernel = concrete_backend::native::ShaderModule;

// KernelModule is a marker trait to bound some Kernel struct types
pub trait AsKernel {}

// empty impl for marker trait
impl AsKernel for Kernel {}
