import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def parse_md_results(path: Path):
    """
    Parses a markdown table of the form:
    | Matrix size | GFLOPS |
    | 512         | 2215.7 ± 269.6 |
    """
    sizes = []
    gflops = []
    errors = []

    row_re = re.compile(
        r"\|\s*(\d+)\s*\|\s*([\d.]+)(?:\s*±\s*([\d.]+))?\s*\|"
    )

    for line in path.read_text().splitlines():
        m = row_re.match(line)
        if not m:
            continue

        sizes.append(int(m.group(1)))
        gflops.append(float(m.group(2)))
        errors.append(float(m.group(3)) if m.group(3) else None)

    return sizes, gflops, errors

def format_algo_name(stem: str) -> str:
    """
    cublas        -> Cublas
    smem_tiling   -> Smem Tiling
    tensor_core_v2 -> Tensor Core V2
    """
    return " ".join(word.capitalize() for word in stem.split("_"))

def plot_results(results_dir: Path, output: str):
    plt.figure(figsize=(8, 5))

    for md_file in sorted(results_dir.glob("*.md")):
        algo_name = format_algo_name(md_file.stem)

        sizes, gflops, errors = parse_md_results(md_file)

        if not sizes:
            print(f"Warning: no data found in {md_file}")
            continue

        plt.plot(
            sizes,
            gflops,
            marker="o",
            label=algo_name,
        )

    plt.xlabel("Matrix size")
    plt.ylabel("GFLOPS")
    plt.title("Kernel Performance Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output == "-":
        plt.show()
    else:
        plt.savefig(output, dpi=150)
        print(f"Saved plot to {output}")


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
