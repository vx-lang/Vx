#!/bin/sh
# Vx toolchain installer.
#
#   curl -fsSL https://vxlang.org/install.sh | sh
#
# Downloads a prebuilt Vx toolchain, checks it against its published SHA-256, and unpacks it
# under ~/.vx. Nothing is installed outside that directory and no command needs root.
#
# The toolchain needs LLVM 22 and the z3 binary present on the system. This script does not
# install them for you -- a script piped into a shell should not be quietly running a package
# manager. It checks for them, and if they are missing it prints the exact command to run.
#
# Environment:
#   VX_VERSION     version to install (default: the latest published release)
#   VX_HOME        install prefix (default: $HOME/.vx)
#   VX_NO_MODIFY_PATH=1   skip the shell-profile PATH suggestion
#   VX_SKIP_CHECKSUM=1    install even if the release publishes no checksum to verify against
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -eu

REPO="vx-lang/Vx"
LLVM_MAJOR="22"
VX_HOME="${VX_HOME:-$HOME/.vx}"

# ------------------------------------------------------------------ output --

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    B=$(printf '\033[1m'); DIM=$(printf '\033[2m')
    RED=$(printf '\033[31m'); GRN=$(printf '\033[32m'); YLW=$(printf '\033[33m')
    R=$(printf '\033[0m')
else
    B=''; DIM=''; RED=''; GRN=''; YLW=''; R=''
fi

say()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "$B" "$R" "$*"; }
warn() { printf '%swarning:%s %s\n' "$YLW" "$R" "$*" >&2; }
err()  { printf '%serror:%s %s\n' "$RED" "$R" "$*" >&2; }

die() { err "$@"; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "this installer needs '$1', which was not found"
}

# ---------------------------------------------------------------- platform --

detect_platform() {
    os=$(uname -s)
    arch=$(uname -m)

    case "$os/$arch" in
        Darwin/arm64)
            TARGET="aarch64-apple-darwin"
            PLATFORM_NAME="macOS (Apple Silicon)"
            ;;
        Linux/x86_64)
            TARGET="x86_64-unknown-linux-gnu"
            PLATFORM_NAME="Linux x86_64"
            ;;
        Darwin/x86_64)
            die "Intel Macs are not supported.
    The Apple accelerator dispatch path and the arm64 runtime assume Apple Silicon.
    Build from source if you need an Intel host: https://vxlang.org/docs/building.html"
            ;;
        Linux/aarch64|Linux/arm64)
            die "Linux on arm64 has no prebuilt toolchain yet.
    Building from source works: https://vxlang.org/docs/building.html"
            ;;
        *)
            die "unsupported platform: $os on $arch
    Prebuilt toolchains exist for macOS on Apple Silicon and Linux x86_64.
    See https://vxlang.org/docs/building.html to build from source."
            ;;
    esac
}

# ------------------------------------------------------------ prerequisites --

# The package-manager line for this platform, so a missing dependency comes with its cure.
deps_hint() {
    case "$TARGET" in
        aarch64-apple-darwin)
            say "    brew install llvm@${LLVM_MAJOR} z3"
            ;;
        x86_64-unknown-linux-gnu)
            say "    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh ${LLVM_MAJOR}"
            say "    sudo apt-get install -y libmlir-${LLVM_MAJOR}-dev mlir-${LLVM_MAJOR}-tools z3 libffi8"
            ;;
    esac
}

# Locate an LLVM $LLVM_MAJOR installation and set LLVM_BIN to its bin directory.
#
# Checked in order of specificity: a version-suffixed llvm-config (how apt.llvm.org installs it),
# Homebrew's versioned and unversioned kegs, then whatever plain `llvm-config` resolves to. The
# last one is checked last on purpose -- on macOS it is usually Xcode's, which is not MLIR-capable.
find_llvm() {
    LLVM_BIN=""

    for candidate in \
        "$(command -v llvm-config-${LLVM_MAJOR} 2>/dev/null || true)" \
        "/usr/lib/llvm-${LLVM_MAJOR}/bin/llvm-config" \
        "/opt/homebrew/opt/llvm@${LLVM_MAJOR}/bin/llvm-config" \
        "/opt/homebrew/opt/llvm/bin/llvm-config" \
        "/usr/local/opt/llvm@${LLVM_MAJOR}/bin/llvm-config" \
        "/usr/local/opt/llvm/bin/llvm-config" \
        "$(command -v llvm-config 2>/dev/null || true)"
    do
        [ -n "$candidate" ] && [ -x "$candidate" ] || continue
        v=$("$candidate" --version 2>/dev/null || echo "")
        case "$v" in
            "${LLVM_MAJOR}".*)
                LLVM_BIN=$("$candidate" --bindir 2>/dev/null || dirname "$candidate")
                LLVM_FOUND_VERSION="$v"
                return 0
                ;;
        esac
    done
    return 1
}

check_prereqs() {
    step "Checking prerequisites"

    missing=0

    if find_llvm; then
        say "    ${GRN}ok${R}  LLVM ${LLVM_FOUND_VERSION}  ${DIM}${LLVM_BIN}${R}"
    else
        err "LLVM ${LLVM_MAJOR} was not found."
        say ""
        say "  Vx lowers through MLIR and runs mlir-translate, opt, llc and clang from LLVM"
        say "  ${LLVM_MAJOR} at compile time. A different major version will not work: the MLIR C API"
        say "  changes between releases."
        say ""
        say "  Install it with:"
        deps_hint
        say ""
        missing=1
    fi

    # The z3 *binary* is executed, not linked. Installing libz3-dev alone is the usual mistake.
    #
    # A warning rather than an error: seam verification is opt-in behind --verify-seams, so a
    # toolchain without z3 compiles and runs everything else perfectly well. Refusing to install
    # over it would block people who will never turn the flag on.
    if command -v z3 >/dev/null 2>&1; then
        say "    ${GRN}ok${R}  z3 $(z3 --version 2>/dev/null | head -1 | sed 's/Z3 version //')"
    else
        say "    ${YLW}--${R}  z3 not found ${DIM}(optional)${R}"
        Z3_MISSING=1
    fi

    [ "$missing" -eq 0 ] || die "install the missing prerequisites above, then run this script again"
}

# --------------------------------------------------------------- download ---

resolve_version() {
    if [ -n "${VX_VERSION:-}" ]; then
        VERSION="$VX_VERSION"
        return
    fi
    step "Resolving the latest release"

    # stderr is discarded on purpose. This request is allowed to fail -- a repository with no
    # published release answers 404 -- and curl would otherwise print its own
    # "(22) The requested URL returned error: 404" before we get a chance to say what that means.
    _releases_json=$(
        $DOWNLOAD "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null
    ) || _releases_json=""

    VERSION=$(
        printf '%s' "$_releases_json" \
            | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
            | head -1
    )

    [ -n "$VERSION" ] || die "no published release found for ${REPO}.

    This usually means no binary release exists yet, rather than anything being wrong on
    your machine. Two ways forward:

      Build from source (works today):
        https://github.com/${REPO}/blob/main/docs/INSTALL.md

      Install a specific version, once one is published:
        curl -fsSL https://vxlang.org/install.sh | VX_VERSION=v0.1.0 sh

    Published releases are listed at https://github.com/${REPO}/releases"
}

setup_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DOWNLOAD="curl -fsSL"
        DOWNLOAD_TO="curl -fL# -o"
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOAD="wget -qO-"
        DOWNLOAD_TO="wget -q --show-progress -O"
    else
        die "this installer needs curl or wget"
    fi
}

verify_checksum() {
    file="$1"; expected="$2"

    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$file" | cut -d' ' -f1)
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
    else
        warn "no sha256sum or shasum on this system; skipping checksum verification"
        return 0
    fi

    if [ "$actual" != "$expected" ]; then
        die "checksum mismatch for $(basename "$file")
    expected  $expected
    actual    $actual
  The download was corrupted or tampered with. Nothing has been installed."
    fi
    say "    ${GRN}ok${R}  sha256 verified"
}

# ---------------------------------------------------------------- install ---

do_install() {
    ARCHIVE="vx-${VERSION}-${TARGET}.tar.gz"
    BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

    TMP=$(mktemp -d 2>/dev/null || mktemp -d -t vx-install)
    trap 'rm -rf "$TMP"' EXIT INT TERM

    step "Downloading Vx ${VERSION} for ${PLATFORM_NAME}"
    $DOWNLOAD_TO "$TMP/$ARCHIVE" "${BASE_URL}/${ARCHIVE}" \
        || die "could not download ${BASE_URL}/${ARCHIVE}
    Check that ${VERSION} publishes an artifact for ${TARGET}:
    https://github.com/${REPO}/releases"

    # Every release publishes a .sha256 beside its archive, so a missing one means something is
    # wrong with the release rather than with this machine. This script is run as `curl | sh`, so
    # it fails closed rather than installing bytes it could not check.
    expected=$($DOWNLOAD "${BASE_URL}/${ARCHIVE}.sha256" 2>/dev/null | cut -d' ' -f1 || true)
    if [ -n "$expected" ]; then
        verify_checksum "$TMP/$ARCHIVE" "$expected"
    elif [ -n "${VX_SKIP_CHECKSUM:-}" ]; then
        warn "no published checksum for ${ARCHIVE}; continuing because VX_SKIP_CHECKSUM is set"
    else
        die "no published checksum for ${ARCHIVE}.

    Every Vx release publishes ${ARCHIVE}.sha256 next to the archive, so this download
    could not be verified and has not been installed.

    Check the release page: https://github.com/${REPO}/releases/tag/${VERSION}
    To install anyway, knowing the archive is unverified: VX_SKIP_CHECKSUM=1"
    fi

    step "Unpacking into ${VX_HOME}"
    DEST="${VX_HOME}/toolchains/${VERSION}"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    tar -xzf "$TMP/$ARCHIVE" -C "$DEST" --strip-components=1

    ln -sfn "$DEST" "${VX_HOME}/current"

    mkdir -p "${VX_HOME}/bin"
    for tool in vxc vx-format vx-opt vx-analyzer; do
        [ -e "${DEST}/bin/${tool}" ] || continue
        ln -sfn "${VX_HOME}/current/bin/${tool}" "${VX_HOME}/bin/${tool}"
    done

    # The toolchain wrapper reads this to find the LLVM tools it shells out to. Written at
    # install time because the location is a property of this machine, not of the release.
    mkdir -p "${DEST}/etc"
    cat > "${DEST}/etc/llvm-env.sh" <<EOF
# Written by the Vx installer. The LLVM ${LLVM_MAJOR} installation found on this machine.
VX_LLVM_BIN="${LLVM_BIN}"
EOF
}

verify_install() {
    step "Verifying the install"

    cat > "$TMP/hello.vx" <<'EOF'
fn main() -> i32 {
  let x : i32 = 21;
  return x * 2;
}
EOF

    if out=$("${VX_HOME}/bin/vxc" --run "$TMP/hello.vx" 2>&1); then
        :
    else
        status=$?
        # --run propagates the program's own exit code, and this program returns 42.
        if [ "$status" -ne 42 ]; then
            say "$out" >&2
            die "the installed compiler could not run a hello-world program"
        fi
    fi

    case "$out" in
        *"exited with code: 42"*) say "    ${GRN}ok${R}  compiled and ran a test program" ;;
        *) say "$out" >&2; die "the installed compiler produced unexpected output" ;;
    esac
}

print_next_steps() {
    say ""
    say "${GRN}${B}Vx ${VERSION} is installed.${R}"
    say ""

    case ":${PATH}:" in
        *":${VX_HOME}/bin:"*)
            say "  ${VX_HOME}/bin is already on your PATH."
            ;;
        *)
            if [ -z "${VX_NO_MODIFY_PATH:-}" ]; then
                say "  Add it to your PATH:"
                say ""
                say "    ${B}export PATH=\"${VX_HOME}/bin:\$PATH\"${R}"
                say ""
                say "  ${DIM}Put that line in ~/.zshrc, ~/.bashrc or your shell's profile to make it stick.${R}"
            fi
            ;;
    esac

    if [ -n "${Z3_MISSING:-}" ]; then
        say ""
        say "  ${YLW}Note:${R} z3 was not found, so ${B}--verify-seams${R} is unavailable."
        say "  Everything else works. To enable it later:"
        deps_hint
    fi

    say ""
    say "  Then try:"
    say ""
    say "    ${B}vxc --run hello.vx${R}      compile and run a program"
    say "    ${B}vxc --help${R}              all options"
    say ""
    say "  Getting started:  ${B}https://vxlang.org/docs/getting-started.html${R}"
    say "  Language tour:    ${B}https://vxlang.org/docs/tour.html${R}"
    say ""
}

# ------------------------------------------------------------------- main ---

main() {
    say ""
    say "${B}Vx${R} — one language, every core"
    say ""

    need_cmd uname
    need_cmd tar
    need_cmd mktemp
    setup_downloader

    detect_platform
    check_prereqs
    resolve_version
    do_install
    verify_install
    print_next_steps
}

main "$@"
