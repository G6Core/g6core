# G6Core

**Beyond the Game. Beyond Limits.**

An independent project building a platform for GTA 6 modding and multiplayer. The PC port comes first; custom servers, modding tools and developer APIs build on it.

[Website and live roadmap](https://g6core.com) · [X](https://x.com/G6Core) · [Discord](https://discord.gg/g6core)

> **Modified version, 2026-11-25.** G6 Runtime is derived from earlier GPL-3.0-or-later work and remains under that licence. Authors, origins and changes are listed in [CREDITS.md](CREDITS.md).

## Status

**Early development: the game runs, but it is not playable yet.** Performance and stability are objectives, not current achievements. Milestones are published on the [live roadmap](https://g6core.com/#roadmap).

First in-game tests, 2026-11-25, on one development machine (NVIDIA RTX 2000 Ada, 16 GB of RAM, 4K display mode):

| Area | State |
|------|-------|
| Boot and first-time setup screens | work, about 30 fps |
| Story scenes and first missions | start and are displayed |
| Controls | respond |
| Frame rate in game | about 2 to 4 fps; 2.5 fps on average in the last minutes measured |
| Rendering | incomplete: some elements are missing |
| Stability | one crash at the first story scene, then a 21-minute session without one after the setting described below |

| Phase | Goal |
|-------|------|
| 1 | PC porting and initial boot (in progress) |
| 2 | Core development |
| 3 | Developer tools |
| 4 | Multiplayer platform |
| 5 | Public release |

## What is in this repository

G6 Runtime is a Windows x64 compatibility runtime. It runs the Xbox Series executable of your own game installation and translates its GPU work to Direct3D 12.

**No game files, code or assets are included, and none are distributed. The Xbox build is not provided: you must have your own.** Shaders are translated on your computer, and saves and shader caches stay there.

## Requirements

- 64-bit Windows and a current Direct3D 12 driver, with mesh shaders and Shader Model 6.6
- A large Windows commit limit: set a bigger page file if your computer has 32 GB of RAM or less
- [Rust](https://rustup.rs/) (the version is pinned in `rust-toolchain.toml`)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**, including the x64 MSVC toolset and a Windows 10 or 11 SDK
- A controller for the game's first-time setup: an Xbox-type (XInput) pad, or a DualSense connected over USB. On a cloud PC, a PlayStation pad may need to be presented as an Xbox pad. Without a recognised pad, the keyboard stands in: Enter is A, Backspace is B, the arrows are the D-pad, Esc is Menu
- About 28 GB of memory was committed during play, so the page file matters

## Build and run

1. In the source folder, run **Setup.cmd** and select the game executable inside your own installation. Keep its accompanying files together. From PowerShell:

   ```powershell
   .\Setup.cmd -GamePath "D:\My Games\Series X\game_scarlett_master.exe"
   ```

   Setup verifies the BCPack format tables, builds the programs and installs them in `%LOCALAPPDATA%\G6 Runtime`.

2. Recommended: run `g6-warm.exe` from that folder to translate shaders ahead of time. This takes several minutes, and has to be done again after the shader translator changes.
3. Open **G6 Runtime** from the Start menu. Logs are in `logs`; `logs\latest.txt` names the latest one.

### If the game stops at the first story scene

While pipelines are still being compiled, a frame can take longer than the game tolerates, and it ends itself (exit code `0xc0000005`). Shorten the time each frame waits for them, then start the game again:

```powershell
[Environment]::SetEnvironmentVariable('G6_PIPELINE_WAIT_MS', '2000', 'User')
```

Some objects may then be missing or appear late. Compiled pipelines are kept, so later runs wait less. Remove the variable to restore the default.

**G6 Runtime (Experimental RT)** starts the game with ray tracing enabled. It is unfinished and can be very slow. See the [ray-tracing notes](docs/fork-ray-tracing.md).

## NVIDIA graphics cards

The console runs some programs in groups of 64 lanes; NVIDIA cards run 32. The translator rewrites those programs, and detects the card by itself.

Development machine: NVIDIA RTX 2000 Ada (16 GB), driver 621.40. The automated tests, shader translation and the first in-game sessions run there. No loss of the graphics device was logged in those sessions.

Known limitations on 32-lane cards:

- A few compute shaders still cannot be translated or are refused by the driver, so some effects can be missing. Most need thread groups spanning several 64-lane waves; others belong to the ray-tracing structure builder.
- Programs that have no other 32-lane form use wave-wide branches as a last resort. This form has caused the graphics device to be lost in earlier testing. Set `G6_WAVE_WIDE=0` to leave those programs out; shaders are then translated again.
- Overlapping pixels are not ordered for the shaders that ask for it, which can cause visual artefacts.

## Development

```powershell
.\Initialize-FormatTables.ps1
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo xtask install-hooks        # asset-policy check on every commit
```

Configuration is described in [g6.example.toml](g6.example.toml). Nothing from the game or any other proprietary source may be committed: see the [asset policy](docs/asset-policy.md).

## Licence and credits

G6 source is licensed under [GPL-3.0-or-later](LICENSE). Authors, origins, contributions and references are listed in [CREDITS.md](CREDITS.md). The BCPack format tables are supplied separately, outside that licence; the setup script downloads and verifies them.

G6Core is an independent project and is not affiliated with, endorsed by, or sponsored by Rockstar Games or Take-Two Interactive.
