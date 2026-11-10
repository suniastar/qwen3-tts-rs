#[cfg(not(feature = "cuda"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/fused_residual_rmsnorm.cu");

    cudaforge::KernelBuilder::new()
        .source_dir("kernels")
        .build_ptx()
        .expect("failed to build cuda kernels");
}
