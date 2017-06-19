extern crate ocl;

use ocl::{ProQue, Buffer, MemFlags};

static KERNEL_SRC: &'static str = r#"
    __kernel void sigmoid(
               __global double const* input,
               __private ulong const input_len,
               __global double* const output)
    {
        for (uint n = 0; n < input_len; n++) {
            output[n] = 1.0 / (1 + exp(-input[n]));
        }
    }
"#;

fn main() {
    let input: Vec<f64> = (-5..5).map(|x| x as f64).collect();

    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(input.len())
        .build().expect("Build ProQue");

    let input_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_only().copy_host_ptr())
        .dims(ocl_pq.dims().clone())
        .host_data(&input)
        .build().unwrap();

    let mut output: Vec<f64> = vec![0.0; input.len()];
    let output_buffer: Buffer<f64> = ocl_pq.create_buffer().unwrap();

    let kern = ocl_pq.create_kernel("sigmoid")
        .unwrap()
        .arg_buf(&input_buffer)
        .arg_scl(input.len())
        .arg_buf(&output_buffer);

    kern.enq().unwrap();

    output_buffer.read(&mut output).enq().unwrap();

    println!("output: {:?}", &output);
}
