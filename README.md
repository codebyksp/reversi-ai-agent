# Reversi/Othello!
## 🧩 Overview

This repository implements a **Reversi/Othello AI agent** for the McGill University **COMP 424 Final Project**. The project explores classic game-playing algorithms — including search, heuristics, and optimization — within the context of a two-player strategy game.
**Project Description & Template** : https://www.overleaf.com/read/vnygbjryrxrt#7b70cb

<p align="center">
  <img src="https://t4.ftcdn.net/jpg/00/90/53/03/240_F_90530312_4Mg3HCsCMW91NVHKWNlBaRo8F5pHhN3c.jpg?w=690&h=388&c=crop">
</p>

## Student Agent (implemented)

**File:** `agents/student_agent.py` — *Version 8.2*
**Short summary (2–3 lines):**

This agent uses iterative deepening with **alpha–beta pruning** (minimax) and a multi-component heuristic that adapts by game phase. The heuristic prioritizes corner control (and avoids C/X-squares), stability (stability map + dynamic weight matrix), mobility, edge control with wedge detection, and capture counts — tuned to emphasize mobility early, balanced play midgame, and piece-count/corner dominance in the late game.


## Setup

To setup the game, clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Playing a game

To start playing a game, we will run the simulator and specify which agents should complete against eachother. To start, several agents are given to you, and you will add your own following the same game interface. For example, to play the game using two copies of the provided random agent (which takes a random action every turn), run the following:

```bash
python simulator.py --player_1 random_agent --player_2 random_agent
```

This will spawn a random game board of size NxN, and run the two agents of class [RandomAgent](agents/random_agent.py). You will be able to see their moves in the console.

## Visualizing a game

To visualize the moves within a game, use the `--display` flag. You can set the delay (in seconds) using `--display_delay` argument to better visualize the steps the agents take to win a game.

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --display
```

## Play on your own!

To take control of one side of the game and compete against the random agent yourself, use a [`human_agent`](agents/human_agent.py) to play the game.

```bash
python simulator.py --player_1 human_agent --player_2 random_agent --display
```

## Autoplaying multiple games

There is some randomness (coming from the initial game setup and potentially agent logic), so to fairly evaluate agents, we will run them against eachother multiple times, alternating their roles as player_1 and player_2, on various board sizes that are selected randomly (between size 6 and 12). The aggregate win % will determine a fair winner. Use the `--autoplay` flag to run $n$ games sequentially, where $n$ can be set using `--autoplay_runs`.

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --autoplay
```

During autoplay, boards are drawn randomly between size `--board_size_min` and `--board_size_max` for each iteration. You may try various ranges for your own information and development by providing these variables on the command-line. However, the defaults (to be used during grading) are 6 and 12, so ensure the timing limits are satisfied for every board in this size range. 

**Notes**

- Not all agents support autoplay (e.g. the human agent doesn't make sense this way). The variable `self.autoplay` in [Agent](agents/agent.py) can be set to `True` to allow the agent to be autoplayed. Typically this flag is set to false for a `human_agent`.
- UI display will be disabled in an autoplay.

## Developing Your Own Agent

To create or modify your own agent:

1. Edit **only** `agents/student_agent.py`
2. Implement the `step` function for your AI logic.
3. Use helper functions from `helpers.py` — do **not** modify other core files.
4. Test your agent with:

   ```bash
   python simulator.py --player_1 student_agent --player_2 random_agent --autoplay
   ```

---

## Optional: Creating Multiple Agents

You can duplicate your agent for experimentation:

```bash
cp agents/student_agent.py agents/second_agent.py
```

Then:

* Update the decorator to `@register_agent("second_agent")`
* Change the class name to `SecondAgent`
* Import it in `agents/__init__.py`
* Run:

  ```bash
  python simulator.py --player_1 student_agent --player_2 second_agent --display
  ```
    
## Auto-Grading Requirements Checklist

* `student_agent.py` is the only modified file
* No extra imports or external libraries
* Decorator remains exactly `@register_agent("student_agent")`
* Runs under 2 seconds per move (configured: `time_limit = 1.90s`)
* Works on all board sizes (6–12)

To test before submission:

```bash
python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
```
## Technical Notes (specific to student_agent.py v8.2)

* **Algorithm:** Iterative deepening minimax with alpha–beta pruning.
* **Move ordering:** Simulation + pre-evaluation of child states to sort moves before search (improves pruning).
* **Heuristics included:** corner control (with C/X penalties), dynamic stability-weighted board, mobility, piece-count (parity), capture-count (late-game emphasis), edge control with wedge detection.
* **Adaptive weights:** Heuristic components scale by game phase (`early`, `middle`, `late`) as determined by piece occupancy.
* **Performance:** Time-limited to ~1.9s per turn to satisfy 2s constraint; iterative deepening attempts deeper searches until timeout.

---

## Future Improvements

* Add **Monte Carlo Tree Search (MCTS)** as an alternative search strategy (especially effective for midgame calibration).
* Implement **transposition table / memoization** for repeated board states to reduce duplicated work.
* Improve **move ordering** using history heuristics or killer moves.
* Tune static/dynamic weight matrices with automated hyperparameter search (grid search or evolutionary methods).
* Add more robust **time-management** (e.g., reserve a small buffer for final move selection).

---
## Full API

```bash
python simulator.py -h       
usage: simulator.py [-h] [--player_1 PLAYER_1] [--player_2 PLAYER_2]
                    [--board_size BOARD_SIZE] [--display]
                    [--display_delay DISPLAY_DELAY]

optional arguments:
  -h, --help            show this help message and exit
  --player_1 PLAYER_1
  --player_2 PLAYER_2
  --board_size BOARD_SIZE
  --display
  --display_delay DISPLAY_DELAY
  --autoplay
  --autoplay_runs AUTOPLAY_RUNS
```

## About

This is a class project for COMP 424, McGill University, Fall 2024 (it was originally forked with the permission of Jackie Cheung).

## License

[MIT](LICENSE)
