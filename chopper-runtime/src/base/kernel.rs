// pub type Kernel = Vec<u32>;
pub type KernelByteCode = Vec<u32>;

pub trait AsKernel {}

pub type Kernel = concrete_backend::native::ShaderModule;
impl AsKernel for Kernel {}
