# Releasing rapidocr 5.x

The workflow is `.github/workflows/build-wheels.yml`.

It builds CPython 3.9–3.13 wheels inside the official manylinux2014 images
(devtoolset GCC 10, glibc 2.17 / `manylinux_2_17`) and an sdist with
`python -m build --sdist`. x86_64 uses `ubuntu-latest`. aarch64 uses the
native `ubuntu-24.04-arm` runner. If that job never starts, the same
aarch64 build runs under QEMU on `ubuntu-latest`. After `auditwheel repair`,
the workflow rejects a wheel whose filename and `auditwheel show` output are
not `manylinux2014` or `manylinux_2_17`.

Triggers:

- push to `V5_NG`
- pull request targeting `V5_NG`
- tag `v5*` (for example `v5.0.0a1`)
- Actions → Run workflow

The wheel build passes `-DPPOCR_ENABLE_VULKAN=OFF`. Shaders stay out of the
published library, so the manylinux image does not need glslang and the
wheel does not gain a `libvulkan` dependency. A Vulkan-enabled build is the
source default when `glslangValidator` and `third_party/Vulkan-Headers` are
present; `.github/workflows/vulkan-lavapipe.yml` runs that build on lavapipe.

The wheel and sdist files are uploaded as the `rapidocr-dist` artifact,
with `SHA256SUMS`. PyPI publishing and the GitHub Release run only for a
push of a `v5*` tag. They do not run for a branch push, a pull request, or a
manual run from a branch.

`5.0.0a1` is a pre-release, so `pip install rapidocr` on PyPI continues to
select the 3.x line until a final 5.x release exists. Install a pre-release
with an explicit version.

## Trusted publisher

Daniel configures this once on PyPI and GitHub. The workflow already
requests `id-token: write` and uses the GitHub Environment named `pypi`.

1. On the `rapidocr` project at PyPI, open Publishing → Add a new pending
   publisher (or an ordinary publisher if the project already exists).
2. Owner: `RapidAI`. Repository: `RapidOCR`. Workflow name:
   `build-wheels.yml`. Environment name: `pypi`.
3. On GitHub, open Settings → Environments and create an environment named
   `pypi`. A required reviewer is optional. Do not put a PyPI password or
   API token in the repository.
4. Settings → Actions → General → Workflow permissions must allow Actions
   to request the `contents: write` permission, or the GitHub Release step
   cannot attach files. The PyPI step only needs the `id-token` permission
   that the publish job already sets.
5. The `ubuntu-24.04-arm` label has to be offered to this repository. Public
   repositories normally have it. If GitHub never assigns a runner, the
   QEMU job builds aarch64 instead.

## Trigger a release

From `V5_NG`, with the version in `pyproject.toml` already set:

```bash
git tag v5.0.0a1
git push origin v5.0.0a1
```

That push runs the wheel build, then publishes `dist/*.whl` and the sdist to
PyPI and opens a GitHub Release with those files and `SHA256SUMS`. Re-running
a tag that is already on PyPI skips the existing file (`skip-existing`).
