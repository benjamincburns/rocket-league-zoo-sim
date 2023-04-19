# rocket-league-zoo-sim
A fork of [AechPro/rocket-league-gym-sim](https://github.com/AechPro/rocket-league-gym-sim) (aka RLGym-sim), but updated to conform to the new[PettingZoo](https://pettingzoo.farama.org/) environment interface standard. RLGym-sim is a version of [RLGym](https://www.rlgym.org) that was forked to give existing users of RLGym an drop-in replacement library that makes use of [RocketSim](https://github.com/ZealanL/RocketSim) simulator, rather than the Rocket League client.

## FOREWORD
This project was originally built as a TEMPORARY STOP-GAP to enable use of RocketSim while RLGym 2.0 is in development. As I intend to track upstream changes in this fork, neither Aech or I provide any guarantees that it is bug-free or that either of us will not make breaking changes to this project in the future.

This project requires you to install [python bindings](https://github.com/mtheall/RocketSim/tree/python-dev) for RocketSim from a separate project, which will require c++20 build tools. Further, you will need to acquire assets from a copy of Rocket League that you own with an asset dumper. Neither Aech or I will walk you through any of this process. The necessary links and basic instructions are listed below. If you cannot follow those, please don't bother either of us.

## INSTALLATION
1. You will need c++20 build tools.
2. Build RocketSim and install the Python bindings via `pip install git+https://github.com/mtheall/RocketSim@feb9f3e50b9526461c142541a8f51e63764014e8` 
3. Install this project with pip via `pip install git+https://github.com/AechPro/rocket-league-gym-sim@main`
4. Build and run the [asset dumper](https://github.com/ZealanL/RLArenaCollisionDumper)
5. Move the dumped assets to the top level of your project directory

## USAGE
With respect to the config classes (e.g. `RewardFunction`, `ObsBuilder`, `StateSetter`, etc), this project acts as a drop-in replacement for RLGym and it can be used in exactly the same way. With respect to the environment interface that is consumed by your learning framework, this library has several breaking changes from RLGym, as it has been updated to conform to the [PettingZoo `ParallelEnv` API](https://pettingzoo.farama.org/api/parallel/).

RLZoo-sim makes use of a subset of the RLGym environment initialization arguments (arguments to the `make` function), as described below.

All variables having to do with the game client have been removed from the `make` function. For example, `rlgymnasium_sim.make(use_injector=True)` will fail because there is no injector. The following is a list of all removed `make` variables:
- `use_injector`
- `game_speed`
- `auto_minimize`
- `force_paging`
- `launch_preference`
- `raise_on_crash`
- `self_play`

Thanks to the flexibility of the simulator, the following additional variables have been added as arguments to `make`:
- `copy_gamestate_every_step`: Leave this alone for the default behavior. Setting this to `True` will no longer return a new `GameState` object at every call to `step`, which is substantially faster. However, if you need to track data from the game state or its member variables over time, you will need to manually copy all relevant `GameState`, `PlayerData`, and `PhysicsObject` data at each `step`.
- `dodge_deadzone`: Sets the threshold value that `pitch` must meet in order for a dodge to occur when jumping in the air.

## KNOWN ISSUES
- Setting the gravity and boost consumption values through `update_settings()` does not work and is not supported.
- A variety of classes in `rlgym_utils` such as `SB3MultipleInstanceEnv` imports the `rlgym` library to build environments, so you will need to replace those imports yourself and remove the misc launch options listed above. You may need to make other changes to rlgym_tools to accommodate this library having migrated to the PettingZoo API.
- the `PlayerData` objects do not track `match_saves` or `match_shots` yet.

## VERSIONING

Unlike the projects from which this is forked, this project will endeavour to make use of the [Semantic Versioning 2.0.0](https://semver.org/#semantic-versioning-200) versioning standard. This means that until a stable 1.0.0 release is made, any feature additions _or_ breaking changes will increment the minor version number, and any bug fixes will increment the patch number.
