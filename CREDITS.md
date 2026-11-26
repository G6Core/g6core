# Credits and origins

G6 Runtime is a modified version of earlier work released under GPL-3.0-or-later. This copy remains under the same licence; see [LICENSE](LICENSE).

**Modified on 2026-11-25:** the names of the project, crates, programs, environment variables, folders and shortcuts were changed to G6. The historical upstream README was removed and its credits were gathered here. The format-table script now reads its source address from this file.

**Modified on 2026-11-26:** changes for the Series S mode on NVIDIA cards, listed below.

## Where this code comes from

- **VIctory**, by Marvisak and its contributors, is the original runtime and the base of everything here: the loader, the runtime services, the Direct3D 12 back end and the shader recompiler. GPL-3.0-or-later. [Marvisak/VIctory](https://github.com/Marvisak/VIctory)
- **fxceless/victory-fork**, a fork of VIctory. [Repository](https://github.com/fxceless/victory-fork)
- **victory-amd**, a fork of victory-fork. [myokex/victory-amd](https://github.com/myokex/victory-amd) (at `070df42`) · [JoshuaO22/victory-amd](https://github.com/JoshuaO22/victory-amd)
- **victory-world line**, by ZSlayerHQ and its contributors, provided the rendering baseline, including the Radeon work from triippywhale and asdaasdsaddf. [victory-world-runtime](https://github.com/ZSlayerHQ/victory-world-runtime) · [Integrated revision](https://github.com/ZSlayerHQ/victory-world-runtime/commit/99f8834aa16b98d7f3943b8de262a8d4857bda7b)
- **DexNom Runtime**, the fork this copy was renamed from. [DexNom/VIctory](https://github.com/DexNom/VIctory)

## Contributions taken in or reworked

- The guard over all the views of a page is code of [victory-fork #115](https://github.com/fxceless/victory-fork/pull/115) by Spexivus, with [its correction](https://github.com/Spexivus/victory-fork/pull/1) by joroyt, and a part of [VIctory #1122](https://github.com/Marvisak/VIctory/pull/1122) by tightfall.
- The second look at textures is fxceless's idea (victory-fork `a803c87`), written anew.
- The loading waits in `crates/g6-gpu/src/pipeline/make.rs` are ported from `joshuu/victory-amd` commit `4436f8f1`.
- The texture catch-up pace in `crates/g6-gpu/src/draw.rs` follows `joshuu/victory-amd`.
- Series S support, ported on 2026-11-26 from [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) and adapted:
  - reading the AMD upscaler's precompiled programs (`ACCC`), with the descriptor and instruction forms they need (commit `66afb2b`, which credits VIctory `0b35b7f`);
  - parked scalars, "some lane is left" and one-channel counter stores (part of commit `389db9e`);
  - one lane read after a whole-wave section (`wave.rs` and its use, part of commit `e5841a0`);
  - two presented planes composed into one picture (commit `85a663d`, from codec-d/cool_fox by Graham, commits `641835a4` and `581b0cb9`).
- Other parts were prompted by pull requests and issues of VIctory and victory-fork, with their code written anew.
- Earlier forks report that their changes were developed with [Claude Code](https://claude.com/claude-code).

## References

- **XCL**, by ivall and its contributors (GPL-2.0-only), served as a reference for how the same title behaves under another compatibility layer. Fixes inspired by it were written anew. [ivall/xcl](https://github.com/ivall/xcl) · [xcl-dev/xcl](https://github.com/xcl-dev/xcl)

Third-party notices for code derived from other projects are kept in the headers of the files concerned.

## BCPack format tables

The upstream project supplies the BCPack format tables as separate data outside the GPL licence. They are not part of this repository. `Initialize-FormatTables.ps1` downloads them from the address below and checks their SHA-256 before use.

Format tables address: https://raw.githubusercontent.com/Marvisak/VIctory/2849477415a46f5a079c85cf3e64a3dff32f713c/crates/victory-formats/src/bcpack-tables.bin
