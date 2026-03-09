#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./build.sh [options]

Builds SuperLU_DIST, hypre, and the local mgr driver using CMake under
./external.

Options:
  -b, --build-type TYPE   CMake build type (default: Release)
  -j, --jobs N            Parallel build jobs (default: nproc)
      --clean             Remove ./external/src, ./external/build, and ./external/install first
      --refresh-sources   Force-fetch requested source refs even if cached locally
      --hypre-tag TAG     hypre git tag/branch (default: v3.1.0)
      --superlu-tag TAG   superlu_dist git tag/branch (default: v9.2.1)
  -h, --help              Show this help text
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${SCRIPT_DIR}/external"
SRC_DIR="${EXTERNAL_DIR}/src"
BUILD_DIR="${EXTERNAL_DIR}/build"
INSTALL_DIR="${EXTERNAL_DIR}/install"

BUILD_TYPE="Release"
JOBS="$(nproc)"
DO_CLEAN=0
FORCE_REFRESH=0

HYPRE_REPO_URL="https://github.com/hypre-space/hypre.git"
HYPRE_TAG="v3.1.0"

SUPERLU_REPO_URL="https://github.com/xiaoyeli/superlu_dist.git"
SUPERLU_TAG="v9.2.1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build-type)
      BUILD_TYPE="${2:-}"
      shift 2
      ;;
    -j|--jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --refresh-sources)
      FORCE_REFRESH=1
      shift
      ;;
    --hypre-tag)
      HYPRE_TAG="${2:-}"
      shift 2
      ;;
    --superlu-tag)
      SUPERLU_TAG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${BUILD_TYPE}" ]]; then
  echo "Error: --build-type requires a non-empty value" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "Error: cmake not found" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git not found" >&2
  exit 1
fi

if [[ ${DO_CLEAN} -eq 1 ]]; then
  rm -rf "${SRC_DIR}" "${BUILD_DIR}" "${INSTALL_DIR}"
fi

mkdir -p "${SRC_DIR}" "${BUILD_DIR}" "${INSTALL_DIR}"

MPICC_BIN="${MPICC:-$(command -v mpicc || true)}"
MPICXX_BIN="${MPICXX:-$(command -v mpicxx || true)}"
MPIFC_BIN="${MPIFC:-$(command -v mpifort || command -v mpif90 || true)}"

if [[ -z "${MPICC_BIN}" || -z "${MPICXX_BIN}" ]]; then
  echo "Error: MPI compilers not found (need mpicc and mpicxx)." >&2
  exit 1
fi

if [[ -z "${MPIFC_BIN}" ]]; then
  echo "Warning: MPI Fortran compiler not found; trying to continue without one."
fi

clone_or_update_repo() {
  local url="$1"
  local tag="$2"
  local dst="$3"
  local dirty=0
  local current_commit=""
  local target_commit=""
  local repo_name=""

  repo_name="$(basename "${dst}")"

  if [[ -d "${dst}/.git" ]]; then
    if ! git -C "${dst}" diff --quiet || ! git -C "${dst}" diff --cached --quiet; then
      dirty=1
    fi

    current_commit="$(git -C "${dst}" rev-parse HEAD)"
    target_commit="$(git -C "${dst}" rev-parse -q --verify "${tag}^{commit}" || true)"

    if [[ ${FORCE_REFRESH} -eq 1 ]]; then
      echo "==> Refreshing ${repo_name} ref ${tag}"
      git -C "${dst}" fetch --depth 1 origin "${tag}"
      target_commit="$(git -C "${dst}" rev-parse FETCH_HEAD)"
    elif [[ -z "${target_commit}" ]]; then
      echo "==> Fetching ${repo_name} ref ${tag}"
      git -C "${dst}" fetch --depth 1 origin "${tag}"
      target_commit="$(git -C "${dst}" rev-parse FETCH_HEAD)"
    fi

    if [[ -n "${target_commit}" && "${current_commit}" == "${target_commit}" ]]; then
      echo "==> Using cached ${repo_name} ref ${tag}"
      return 0
    fi

    if [[ ${dirty} -eq 1 ]]; then
      echo "Error: ${dst} has local changes; cannot switch to ${tag}." >&2
      echo "Use --clean to rebuild from a fresh source checkout." >&2
      return 1
    fi

    git -C "${dst}" checkout --detach "${target_commit}"
  else
    echo "==> Cloning ${repo_name} ref ${tag}"
    git clone --branch "${tag}" --depth 1 "${url}" "${dst}"
  fi
}

find_one_lib() {
  local libdir="$1"
  local stem="$2"
  local found=""
  for pat in "lib${stem}.so" "lib${stem}.dylib" "lib${stem}.a"; do
    if [[ -f "${libdir}/${pat}" ]]; then
      found="${libdir}/${pat}"
      break
    fi
  done
  printf '%s' "${found}"
}

SUPERLU_SRC="${SRC_DIR}/superlu_dist"
SUPERLU_BUILD="${BUILD_DIR}/superlu_dist"
SUPERLU_INSTALL="${INSTALL_DIR}/superlu_dist"

HYPRE_SRC="${SRC_DIR}/hypre"
HYPRE_BUILD="${BUILD_DIR}/hypre"
HYPRE_INSTALL="${INSTALL_DIR}/hypre"

DRIVER_BUILD="${BUILD_DIR}/mgr_driver"

echo "==> Preparing sources"
clone_or_update_repo "${SUPERLU_REPO_URL}" "${SUPERLU_TAG}" "${SUPERLU_SRC}"
clone_or_update_repo "${HYPRE_REPO_URL}" "${HYPRE_TAG}" "${HYPRE_SRC}"

echo "==> Configuring SuperLU_DIST (${SUPERLU_TAG})"
cmake -S "${SUPERLU_SRC}" -B "${SUPERLU_BUILD}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${SUPERLU_INSTALL}" \
  -DCMAKE_C_COMPILER="${MPICC_BIN}" \
  -DCMAKE_CXX_COMPILER="${MPICXX_BIN}" \
  ${MPIFC_BIN:+-DCMAKE_Fortran_COMPILER="${MPIFC_BIN}"} \
  -DBUILD_SHARED_LIBS=ON \
  -DTPL_ENABLE_PARMETISLIB=OFF \
  -Denable_tests=OFF \
  -Denable_examples=OFF

echo "==> Building SuperLU_DIST"
cmake --build "${SUPERLU_BUILD}" --parallel "${JOBS}"
cmake --install "${SUPERLU_BUILD}"

SUPERLU_INCLUDE_DIR="${SUPERLU_INSTALL}/include"
SUPERLU_LIB_DIR="${SUPERLU_INSTALL}/lib"
if [[ ! -d "${SUPERLU_LIB_DIR}" ]]; then
  SUPERLU_LIB_DIR="${SUPERLU_INSTALL}/lib64"
fi
SUPERLU_LIB="$(find_one_lib "${SUPERLU_LIB_DIR}" "superlu_dist")"
if [[ -z "${SUPERLU_LIB}" ]]; then
  echo "Error: could not find libsuperlu_dist in ${SUPERLU_LIB_DIR}" >&2
  exit 1
fi

echo "==> Configuring hypre (${HYPRE_TAG}) with SuperLU_DIST"
cmake -S "${HYPRE_SRC}/src" -B "${HYPRE_BUILD}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${HYPRE_INSTALL}" \
  -DCMAKE_C_COMPILER="${MPICC_BIN}" \
  -DCMAKE_CXX_COMPILER="${MPICXX_BIN}" \
  ${MPIFC_BIN:+-DCMAKE_Fortran_COMPILER="${MPIFC_BIN}"} \
  -DBUILD_SHARED_LIBS=ON \
  -DHYPRE_ENABLE_DSUPERLU=ON \
  -DTPL_DSUPERLU_INCLUDE_DIRS="${SUPERLU_INCLUDE_DIR}" \
  -DTPL_DSUPERLU_LIBRARIES="${SUPERLU_LIB}"

echo "==> Building hypre"
cmake --build "${HYPRE_BUILD}" --parallel "${JOBS}"
cmake --install "${HYPRE_BUILD}"

HYPRE_INCLUDE_DIR="${HYPRE_INSTALL}/include"
HYPRE_LIB_DIR="${HYPRE_INSTALL}/lib"
if [[ ! -d "${HYPRE_LIB_DIR}" ]]; then
  HYPRE_LIB_DIR="${HYPRE_INSTALL}/lib64"
fi
HYPRE_LIB="$(find_one_lib "${HYPRE_LIB_DIR}" "HYPRE")"
if [[ -z "${HYPRE_LIB}" ]]; then
  echo "Error: could not find libHYPRE in ${HYPRE_LIB_DIR}" >&2
  exit 1
fi

echo "==> Configuring mgr-amgf driver"
cmake -S "${SCRIPT_DIR}" -B "${DRIVER_BUILD}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_C_COMPILER="${MPICC_BIN}" \
  -DHYPRE_INCLUDE_DIRS="${HYPRE_INCLUDE_DIR}" \
  -DHYPRE_LIBRARIES="${HYPRE_LIB}"

echo "==> Building mgr-amgf driver"
cmake --build "${DRIVER_BUILD}" --parallel "${JOBS}"

cat <<EOF
Build complete.
  SuperLU_DIST install: ${SUPERLU_INSTALL}
  hypre install:        ${HYPRE_INSTALL}
  mgr driver build:     ${DRIVER_BUILD}/mgr_driver
EOF
