import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt


ARTICLE_ORDER = [
    "naive",
    "memory_coalescing",
    "smem_tiling",
    "1d_blocktiling",
    "2d_blocktiling",
    "tf32_tensor_block128x128",
    "tf32_tensor_ptx_async",
    "tf32_tensor_ptx_async_warp32x64",
    "tf32_tensor_ptx_async_block128x256_warp32x64",
    "tf32_tensor_ptx_async_block128x256_warp32x64_aligned",
    "cublas",
]

ARTICLE_LABELS = {
    "naive": "Naive CUDA",
    "memory_coalescing": "Coalesced Threads",
    "smem_tiling": "Shared Memory Tiling",
    "1d_blocktiling": "1D Block Tiling",
    "2d_blocktiling": "2D Register Tiling",
    "tf32_tensor_block128x128": "TF32 Tensor Core",
    "tf32_tensor_ptx_async": "PTX MMA + Async Copy",
    "tf32_tensor_ptx_async_warp32x64": "Wider Warp Tile",
    "tf32_tensor_ptx_async_block128x256_warp32x64": "Wider CTA Tile",
    "tf32_tensor_ptx_async_block128x256_warp32x64_aligned": "Aligned Fast Path",
    "cublas": "cuBLAS TF32 Reference",
}


def format_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.1e}"


def parse_optional_float(value):
    if value.lower() in {"", "n/a", "na", "nan"}:
        return None
    return float(value)


def parse_md_results(path: Path):
    """
    Parses old and new markdown result tables.
    """
    sizes = []
    gflops = []
    gflops_errors = []
    max_abs_errors = []
    rel_linf_errors = []
    rel_l2_errors = []

    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue

        cols = [col.strip() for col in line.strip().strip("|").split("|")]
        if not cols or not cols[0].isdigit():
            continue

        gflops_parts = re.findall(r"[\d.]+", cols[1])
        sizes.append(int(cols[0]))
        gflops.append(float(gflops_parts[0]))
        gflops_errors.append(float(gflops_parts[1]) if len(gflops_parts) > 1 else None)
        max_abs_errors.append(parse_optional_float(cols[2]) if len(cols) > 2 else None)
        rel_linf_errors.append(parse_optional_float(cols[3]) if len(cols) > 3 else None)
        rel_l2_errors.append(parse_optional_float(cols[4]) if len(cols) > 4 else None)

    return sizes, gflops, gflops_errors, max_abs_errors, rel_linf_errors, rel_l2_errors

def format_algo_name(stem: str) -> str:
    """
    cublas        -> Cublas
    smem_tiling   -> Smem Tiling
    tensor_core_v2 -> Tensor Core V2
    """
    return " ".join(word.capitalize() for word in stem.split("_"))


def article_label(stem: str) -> str:
    return ARTICLE_LABELS.get(stem, format_algo_name(stem))


def ordered_result_files(results_dir: Path):
    files_by_stem = {path.stem: path for path in results_dir.glob("*.md")}
    ordered = [files_by_stem.pop(stem) for stem in ARTICLE_ORDER if stem in files_by_stem]
    ordered.extend(files_by_stem[stem] for stem in sorted(files_by_stem))
    return ordered

def error_output_path(output: str):
    if output == "-":
        return "-"

    output_path = Path(output)
    suffix = output_path.suffix or ".png"
    return output_path.with_name(f"{output_path.stem}_error{suffix}")


def plot_results(results_dir: Path, output: str):
    fig, perf_ax = plt.subplots(figsize=(16, 9))
    error_fig, error_ax = plt.subplots(figsize=(16, 7))
    plotted_error = False

    for md_file in ordered_result_files(results_dir):
        algo_name = article_label(md_file.stem)

        sizes, gflops, gflops_errors, max_abs_errors, rel_linf_errors, rel_l2_errors = parse_md_results(md_file)

        if not sizes:
            print(f"Warning: no data found in {md_file}")
            continue

        latest_error = next((value for value in reversed(rel_linf_errors) if value is not None), None)
        label = f"{algo_name} (err {format_metric(latest_error)})"

        perf_ax.plot(
            sizes,
            gflops,
            marker="o",
            label=label,
        )

        if any(value is not None for value in rel_linf_errors):
            plotted_error = True
            error_points = [(size, value) for size, value in zip(sizes, rel_linf_errors) if value is not None]
            error_ax.plot(
                [size for size, _ in error_points],
                [value for _, value in error_points],
                marker="o",
                label=label,
            )

    perf_ax.set_ylabel("GFLOPS")
    perf_ax.set_xlabel("Matrix size")
    perf_ax.set_title("Kernel Performance Comparison")
    perf_ax.grid(True, alpha=0.3)
    perf_ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")
    fig.tight_layout(rect=[0, 0, 0.72, 1])

    error_ax.set_xlabel("Matrix size")
    error_ax.set_ylabel("Rel L-inf err")
    error_ax.set_title("Kernel Relative Error vs Naive FP32 Reference")
    error_ax.grid(True, alpha=0.3)
    if plotted_error:
        error_ax.set_yscale("log")
        error_ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")
    else:
        error_ax.text(0.5, 0.5, "No error metrics found", ha="center", va="center", transform=error_ax.transAxes)
    error_fig.tight_layout(rect=[0, 0, 0.72, 1])

    if output == "-":
        plt.show()
    else:
        fig.savefig(output, dpi=150)
        error_path = error_output_path(output)
        error_fig.savefig(error_path, dpi=150)
        print(f"Saved plot to {output}")
        print(f"Saved error plot to {error_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <results_dir> <output.png | ->")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output = sys.argv[2]

    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    plot_results(results_dir, output)
