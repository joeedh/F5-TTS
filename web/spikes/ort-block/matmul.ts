// Hand-written WGSL baseline for D3: tiled fp32 matmul C[M,N] = A[M,K] * B[K,N].
// 64x64 output tile per workgroup of 16x16 threads, 4x4 outputs per thread, K tiles of 16.
// Needs M and N to be multiples of 64 and K a multiple of 16 (true for every spike shape).

const TILE = 64;

const SHADER = /* wgsl */ `
struct Dims { M: u32, N: u32, K: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> As: array<array<f32, 64>, 16>;  // As[k][m]
var<workgroup> Bs: array<array<f32, 64>, 16>;  // Bs[k][n]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row0 = wg.y * 64u;
  let col0 = wg.x * 64u;
  let tid = l.y * 16u + l.x;
  var acc: array<array<f32, 4>, 4>;
  for (var k0 = 0u; k0 < dims.K; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      let ar = idx / 16u;
      let ac = idx % 16u;
      As[ac][ar] = A[(row0 + ar) * dims.K + k0 + ac];
      let br = idx / 64u;
      let bc = idx % 64u;
      Bs[br][bc] = B[(k0 + br) * dims.N + col0 + bc];
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) {
        a[i] = As[k][l.y + 16u * i];
        b[i] = Bs[k][l.x + 16u * i];
      }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
          acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      C[(row0 + l.y + 16u * i) * dims.N + col0 + l.x + 16u * j] = acc[i][j];
    }
  }
}
`;

export interface MatmulResult {
  output: Float32Array;
  timesMs: number[];
}

export async function runWgslMatmul(
  device: GPUDevice,
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
  warmup: number,
  runs: number,
): Promise<MatmulResult> {
  if (m % TILE || n % TILE || k % 16) throw new Error(`shape ${m}x${k}x${n} not tile aligned`);
  const storage = (data: Float32Array): GPUBuffer => {
    const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  };
  const bufA = storage(a);
  const bufB = storage(b);
  const bufC = device.createBuffer({ size: m * n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([m, n, k, 0]));
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [bufA, bufB, bufC, dims].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const dispatch = (): void => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(n / TILE, m / TILE);
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  for (let i = 0; i < warmup; i++) dispatch();
  await device.queue.onSubmittedWorkDone();
  const timesMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    dispatch();
    await device.queue.onSubmittedWorkDone();
    timesMs.push(performance.now() - t0);
  }

  const read = device.createBuffer({ size: m * n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(bufC, 0, read, 0, m * n * 4);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const output = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  for (const buf of [bufA, bufB, bufC, dims, read]) buf.destroy();
  return { output, timesMs };
}
