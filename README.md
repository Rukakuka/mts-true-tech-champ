# mts-true-tech-champ

## Before usage:


Clone the repo into `mts-true-tech-champ` folder, cd to it and add `mts-true-tech-champ` pythonpath:

```
cd ~
git clone https://github.com/Rukakuka/mts-true-tech-champ.git
cd ~/mts-true-tech-champ
source .env
```

### Common solutions

The draft solutions are located in [solutions_common](solutions_common/) directory. To run any of the solution, run the following command pattern in the root of this repo:

```bash
PYTHONPATH=. python3 solutions_common/task_1.py --api_token <YOUR TOKEN>
```

### Lightweight ascii simulator:


Accepts the same REST api calls like GUI simulator, thus it **CANNOT WORK SIMULATENEOUSLY WITH GUI SIMULATOR**

launch it in a separate window, and launch your code in another one

```
python3 lib/ascii/simulator.py
```

It will clear screen and print ascii representation of the maze:

```
  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  
│           │               │                   │                │
      ──          ──      ──      ──      ──  ──      ──          
│       │   │       │           │   │               │   │   │    │
              ──      ──  ──  ──      ──  ──  ──  ──              
│   │   │   │   │   │                               │   │   │    │
                          ──          ──  ──  ──                  
│   │   │   │   │   │       │   │   │           │   │   │   │    │
                      ──  ──              ──                      
│   │   │       │               │   │   │       │       │   │    │
          ──      ──  ──  ──      ──          ──  ──  ──          
│   │       │   │               │       │                   │    │
      ──      ──      ──  ──  ──      ──  ──  ──  ──  ──  ──      
│   │   │           │   │           │           │                │
          ──  ──  ──          ──  ──          ──      ──  ──  ──  
│   │                   │           │   │   │       │            │
          ──  ──  ──  ──                          ──      ──      
│   │   │       │           │       │   │       │   │   │   │    │
                      ──  ──  ──  ──      ──  ──                  
│   │       │   │   │       │       │           │   │       │    │
      ──  ──          ──              ──  ──              ──      
│               │       │   │   │       │       │       │        │
  ──      ──  ──  ──                  ──      ──      ──          
│       │       │       │       │       │   │       │   │   │    │
      ──              ──      ──  ──              ──          ──  
│   │       │       │       │       │       │           │        │
          ──  ──  ──      ──  ──      ──  ──  ──  ──  ──  ──      
│   │       │       │       │                       │            │
      ──              ──          ──  ──  ──  ──          ──  ──  
│       │   │   │       │   │       │           │   │            │
  ──  ──              ──      ──          ──          ──  ──      
│ ⇧ robot here  │                   │       │                    │
  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  ──  
```

### Launch tests:

#### Test task 1
```
test/test_task1.py --help

usage: test_task1.py [-h] [--no-ui] [--maze-path MAZE_PATH] [--test-script TEST_SCRIPT]

options:
  -h, --help                    show this help message and exit
  --no-ui                       supress ascii rendering
  --maze-path MAZE_PATH         path to json with test mazes
  --test-script TEST_SCRIPT     path to .py tested script
```  

Example usage:

```
python3 test/test_task1.py --test-script junk/task_1_backtrace.py --maze-path test/mazes.json --no-ui
```

Test suite expects api call `/api/v1/matrix/send` **disregarding** your test script termination. It will print **Test OK** if expected and your script output mazes are equal, or **Test FAIL** with diff between expected and actual maze.

#### Test task 2

```
test/test_task2.py --help

usage: test_task2.py [-h] [--no-ui] [--maze-path MAZE_PATH] [--test-script TEST_SCRIPT]

options:
  -h, --help            show this help message and exit
  --no-ui
  --maze-path MAZE_PATH
  --test-script TEST_SCRIPT
```

Example usage:

```
python3 test/test_task2.py  --test-script solutions_common/task_2.py --maze-path junk/debug_maze.json --no-ui
```

Test suite expects api call `/api/v1/maze/reset` 2 times and expects the robot to have at least 1 completed run and at max
3 completed runs. Test expects tthe tested script to perform maze reset by itself.

It will print **Test OK** if conditions matches, or **Test FAIL** and the reason with some details.

Example output in case if tested script fails to make 2 restart calls:
```
maze/restart api calls is 1 times, expected 2 times
Test #1 FAIL
```

In case of success, time of each run is given:
```
Times: attempt1=5.158s, attempt2=5.146s, attempt3=5.137s
Test #1 OK!
```

### Junk:

Just code with no rules
