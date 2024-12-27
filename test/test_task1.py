from lib.ascii.app import sim
import os
import time
import sys
import json
import argparse
import subprocess
import multiprocessing
import signal

GREEN = '\033[32m'
RESET = '\033[0m'
RED = '\033[31m'


def script_worker(script_path, output_queue):
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        stdout, stderr = process.communicate()

        output = {"stdout": stdout.strip(), "stderr": stderr}
        output_queue.put(output)  # Place the output in the queue

    except Exception as e:
        output_queue.put({"stderr": str(e)})


def run_tested_script(script_path):
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=script_worker, args=(script_path, output_queue))
    process.start()
    process.join()

    result = output_queue.get()
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    if stdout:
        print("Test script stdout:\n", stdout)
    if stderr:
        print("Test script stderr:\n", stderr)

    return stdout


def exit(sim=None):
    os.kill(os.getpid(), signal.SIGTERM)
    if sim is not None:
        sim.terminate()


def get_test1_results(sim):
    return sim.get_score(), sim.get_compare_maze_result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-ui', action='store_true')
    parser.add_argument('--maze-path', type=str, default='test/mazes.json')
    parser.add_argument('--test-script', type=str)

    args = parser.parse_args()

    if args.test_script is None or args.test_script == '':
        print("Test script should be provided")
        exit()

    try:
        with open(args.maze_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at path '{args.maze_path}'")
        exit()
    except json.JSONDecodeError:
        print("Error: JSON decode error. Please check the file for proper JSON format.")
        exit()

    mazes_without_loops = data.get('mazes_without_loops', [])
    mazes_with_loops = data.get('mazes_with_loops', [])
    mazes_with_different_entries = data.get('mazes_with_different_entries', [])

    maze = mazes_without_loops[0]

    all_mazes = mazes_without_loops + mazes_with_loops + mazes_with_different_entries

    sim.reset(maze)
    sim.run(render=True)

    run = 1
    for test_maze in all_mazes:
        maze = test_maze
        sim.reset(maze)

        print("=== Tested script launch ===")
        if args.test_script:
            _ = run_tested_script(args.test_script)
        print("=== Tested script finished ===")
        score, result = get_test1_results(sim)

        if (result is True):
            print(f"{GREEN}Test #{run} OK! Score: {score}{RESET}")
        else:
            print(f"{RED}Test #{run} FAIL! Score: {score}{RESET}")

        time.sleep(1)
        run += 1

    exit()


if __name__ == "__main__":
    main()
