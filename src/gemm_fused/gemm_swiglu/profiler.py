import subprocess
import re

def find_best_gemm_config(M, N, K):
    cmd = [
        "./cutlass_profiler",
        "--operation=Gemm",
        f"--m={M}", f"--n={N}", f"--k={K}",
        "--profiling-iterations=50",
        "--accumulator-type=f32,f32",
        "--warmup-iterations=10"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="third_party/cutlass/build/tools/profiler")
        output = result.stdout

        best_gflops = 0
        best_kernel = ""

        for line in output.split('\n'):
            if "GFLOPs:" in line:
                gflops = float(re.search(r"GFLOPs:\s*([\d.]+)", line).group(1))
                if gflops > best_gflops:
                    best_gflops = gflops
            elif "Operation:" in line and best_gflops > 0:
                kernel_name = line.split("Operation:")[1].strip()
                best_kernel = kernel_name
                break

        print(f"Best kernel: {best_kernel}")
        print(f"Best GFLOPs: {best_gflops}")
        return best_kernel, best_gflops
        
    except Exception as e:
        print(f"Profiler failed: {e}")
        return None, 0

if __name__ == "__main__":
    M, N, K = 2048, 4096, 1000
    kernel, gflops = find_best_gemm_config(M, N, K)