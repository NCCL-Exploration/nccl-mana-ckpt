import argparse
from pathlib import Path
from json import dump, load
from subprocess import run, CompletedProcess, CalledProcessError
from collections import defaultdict
from statistics import median



BENCHMARK_SCRIPTS : dict[str, str] = {
    "nas_is_perlmutter" : "run_perlmutter_ckpt.sh",
    "nas_is_aicr" : "run_aicr_ckpt.sh"
}


def parse_args() -> argparse.Namespace:
    """ Parses the arguments of this script.

    Returns:
        An argparse data structure containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
            prog = "run_benchmarks_median",
            description = "Runs a given benchmark a given amount of times, and displays the median values."
    )
    parser.add_argument(
            "--home",
            type = Path,
            default = Path.home() / "nccl-mana-ckpt",
            help = "Path to nccl-mana-ckpt home."
    )
    parser.add_argument(
            "--target",
            type = str,
            required = True,
            choices = list(BENCHMARK_SCRIPTS.keys()),
            help = "Target benchmark to run."
    )
    parser.add_argument(
            "--runs",
            type = int,
            default = 9,
            help = "Amount of times to run the given benchmark."
    )
    parser.add_argument(
            "--output_path",
            type = Path,
            default = Path.cwd(),
            help = "Path to where the script's JSON output should be written to."
    )
    return parser.parse_args()

def construct_script_path(home_path : Path, target : str) -> Path:
    """ Constructs the path to the script path, given the home path.

    Args:
       home_path: The home path of the user.
       target: The benchmark to run.

    Returns:
        An updated path to the benchmark script to execute.

    Raises:
        ValueError if the target is invalid.
        FileNotFound error if the script file cannot be found.
    """
    target_script : str = BENCHMARK_SCRIPTS.get(target)
    if target_script is None:
        raise ValueError(f"Uknown target {target}.")
    benchmark_path : Path = home_path / "tests" / "nas-benchmarks" / "IS" / target_script
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Resolved path for {benchmark_path} does not exist.")
    return benchmark_path

def construct_output_path(output_path : Path) -> Path:
    """ Construct the path to the output folder. If it does not exist, create it.

    Args:
        output_path: This is the output folder for the JSON summary.

    Returns:
        The path to the output directory.
    """
    output_path.mkdir(parents = True, exist_ok = True)
    return output_path


def calculate_benchmark_medians(script_path : Path, run_count : int) -> dict[str, float]:
    """ Given a script path and a run count, runs that script under sbatch
        the given count amount of times in series.
        Outputs benchmark results as medians.

    Args:
        script_path: The path to the benchmark script to run.
        run_count: The amount of times to run the benchmark script.

    Returns:
        A dictionary of the benchmark measures, to their median value.
    """
    benchmark_outputs : list[Path] = [run_benchmark(script_path, i) for i in range(run_count)]
    benchmark_measures : dict[str, list[float]] = accumulate_results(benchmark_outputs)
    for benchmark_path in benchmark_outputs:
        benchmark_path.unlink(missing_ok = True)
    return compute_medians(benchmark_measures)

def run_benchmark(script_path : Path, cur_iter : int) -> Path:
    """ Given a script path and the current run iterations,
        runs the script under sbatch, and outputs to an indexed file.

    Args:
        script_path: The path to the benchmark script to run.
        cur_iter: The current iteration of the benchmarks to run.

    Returns:
        The path to the output JSON file containing the benchmark results.

    Raises:
        CalledProcessError: If the sbatch run failed.
        FileNotFoundError: If the output file was not found.
    """
    output_path : Path = script_path.parent / f"trial_{cur_iter}.json"
    result : CompletedProcess = run(
            ["sbatch", "--wait", f"--export=OUTPUT={output_path}", str(script_path)],
            capture_output = True,
            check = True
    )
    if not output_path.exists():
        raise FileNotFoundError(f"Trial completed but {output_path} does not exist.")
    return output_path

def accumulate_results(benchmark_outputs : list[Path]) -> dict[str, list[float]]:
    """ Given a list of paths to JSON files, 
        stores their results as a list for each key.

    Args:
        benchmark_outputs: The paths to the benchmark JSON outputs.

    Returns:
        A dictionary from JSON key, to a list of its values.
    """
    benchmark_measures : dict[str, list[float]] = defaultdict(list)
    for benchmark_path in benchmark_outputs:
        with benchmark_path.open("r") as f:
            cur_data : dict[str, float] = load(fp = f)
        for (key, value) in cur_data.items():
            benchmark_measures[key].append(value)
    return benchmark_measures

def compute_medians(benchmark_measures : dict[str, list[float]]) -> dict[str, float]:
    """ Given a dictionary of benchmark keys and multi-run values,
        returns a dictionary of those same keys, but with the medians.

    Args:
        benchmark_measures: The given dictionary of multi-run benchmark results.

    Returns:
        A dictionary from benchmark measure, to its median value across runs.
    """
    return {key: median(values) for (key, values) in benchmark_measures.items()}

def main() -> None:
    """ Runs a given benchmark test some amount of times,
        and outputs the median of the results to a JSON file.
    """
    args : argparse.Namespace = parse_args()
    script_path : Path = construct_script_path(args.home, args.target)
    output_dir : Path = construct_output_path(args.output_path)
    median_values : dict[str, float] = calculate_benchmark_medians(script_path, args.runs) 

    output_file : Path = output_dir / f"{args.target}_{args.runs}_medians.json"
    with output_file.open("w") as f:
        dump(obj = median_values, fp = f, indent = 2)

    print(f"Wrote median results to {output_file}")


if __name__ == "__main__":
    main()
