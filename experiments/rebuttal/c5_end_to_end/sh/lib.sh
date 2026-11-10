#!/usr/bin/env bash
# Shared shell helpers of C5. Sourced after env.sh by every sh/*.sh.
#
#   c5_run_step <step_id> <title> <report.md> <log> <command...>
#       runs the command with its output in the log; when it dies before writing its report,
#       writes a report with the exit code and the log tail. Returns the exit code.
#   c5_push <dataset|all> <step> <report.md>
#       commits ONLY results/rebuttal/c5_end_to_end/ and experiments/rebuttal/c5_end_to_end/
#       and pushes to ${C5_GIT_BRANCH}. Never "git add -A". Skips files over 5 MB, .env files and
#       anything that looks like a key or token. "git pull --rebase" and up to 3 tries; a failed
#       push is written into the report and never stops the experiment (always returns 0).

C5_GIT_PATHS=("results/rebuttal/c5_end_to_end" "experiments/rebuttal/c5_end_to_end")

c5_run_step() {
  local step="$1" title="$2" report="$3" log="$4"
  shift 4
  [[ -f "${report}" ]] && mv -f "${report}" "${report%.md}.prev.md"
  rm -f "${report%.md}.json"
  {
    echo "[c5] $(date '+%F %T') start ${step}: $*"
  } >> "${log}"
  "$@" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  if [[ ! -f "${report}" ]]; then
    {
      echo "# C5 end-to-end QA - ${title}: FAILED before writing its report"
      echo
      echo "Command: \`$*\`"
      echo
      echo "Exit code: ${status}. Last 80 log lines (${log#"${C5_ROOT_DIR}/"}):"
      echo
      echo '```'
      tail -n 80 "${log}" || true
      echo '```'
    } > "${report}"
    [[ "${status}" -eq 0 ]] && status=1
  fi
  echo "[c5] $(date '+%F %T') end ${step}: exit ${status}" >> "${log}"
  return "${status}"
}

# One-line result for the commit message: the first line of <report>.summary.txt, else the
# report's first heading.
c5_summary_line() {
  local report="$1" line=""
  if [[ -f "${report%.md}.summary.txt" ]]; then
    line="$(head -n 1 "${report%.md}.summary.txt")"
  elif [[ -f "${report}" ]]; then
    line="$(grep -m 1 '^# ' "${report}" | sed 's/^# //')"
  fi
  line="$(echo "${line}" | tr -d '\r' | cut -c1-150)"
  [[ -z "${line}" ]] && line="no summary"
  echo "${line}"
}

# Secrets: a short list of key and token shapes (Hugging Face, OpenAI/Anthropic, GitHub, AWS,
# private keys, explicit api_key assignments).
C5_SECRET_RE='(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api[_-]?key|access[_-]?token|secret)["'"'"' ]*[:=][ ]*["'"'"'][A-Za-z0-9_-]{16,})'

c5_lock_acquire() {
  # Both GPU scripts push from the same checkout: serialise every git call on one lock.
  local lockdir="${C5_ROOT_DIR}/.git/c5_push.lock.d" waited=0
  while ! mkdir "${lockdir}" 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
    if [[ "${waited}" -ge 600 ]]; then
      echo "[c5-git] lock held for 10 min; removing a stale lock ${lockdir}"
      rm -rf "${lockdir}"
    fi
  done
  echo "$$" > "${lockdir}/pid"
}

c5_lock_release() {
  rm -rf "${C5_ROOT_DIR}/.git/c5_push.lock.d"
}

c5_push() {
  local dataset="$1" step="$2" report="$3"
  local note="${report%.md}.push.txt"
  local msg="C5 ${dataset} ${step}: $(c5_summary_line "${report}")"
  if [[ "${C5_GIT_PUSH}" != "1" ]]; then
    echo "push disabled (C5_GIT_PUSH=${C5_GIT_PUSH})" > "${note}"
    return 0
  fi
  (
    cd "${C5_ROOT_DIR}" || exit 0
    c5_lock_acquire
    trap c5_lock_release EXIT
    local branch
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    if [[ "${branch}" != "${C5_GIT_BRANCH}" ]]; then
      echo "NOT PUSHED: checkout is on branch '${branch}', expected '${C5_GIT_BRANCH}'." | tee "${note}"
      exit 0
    fi
    # Candidate files: everything under the two C5 folders (ignored files too, e.g. *.log),
    # minus large files, .env files and anything with a secret.
    local max_bytes=$((C5_GIT_MAX_FILE_MB * 1024 * 1024)) f size skipped=()
    local add=()
    while IFS= read -r -d '' f; do
      case "${f}" in
        */.env|*/.env.*|*.env|*/__pycache__/*|*.pyc|*.prev.md.tmp) skipped+=("${f} (excluded type)"); continue ;;
      esac
      size=$(wc -c < "${f}" 2>/dev/null || echo 0)
      if [[ "${size}" -gt "${max_bytes}" ]]; then
        skipped+=("${f} (${size} bytes > ${C5_GIT_MAX_FILE_MB} MB)")
        continue
      fi
      if grep -Eq "${C5_SECRET_RE}" "${f}" 2>/dev/null; then
        skipped+=("${f} (looks like it holds a key or token)")
        continue
      fi
      add+=("${f}")
    done < <(find "${C5_GIT_PATHS[@]}" -type f -print0 2>/dev/null)
    if [[ "${#add[@]}" -eq 0 ]]; then
      echo "nothing to commit" > "${note}"
      exit 0
    fi
    git add -f -- "${add[@]}" >/dev/null 2>&1
    # Commit only the C5 paths, whatever else is staged in this checkout. A pathspec that matches no
    # file known to git makes "git commit -- <paths>" fail, so only folders git knows are passed.
    local known=() d
    for d in "${C5_GIT_PATHS[@]}"; do
      [[ -n "$(git ls-files -- "${d}" | head -n 1)" ]] && known+=("${d}")
    done
    if [[ "${#known[@]}" -eq 0 ]] || git diff --cached --quiet -- "${known[@]}"; then
      echo "no changes in the C5 folders since the last commit" > "${note}"
      exit 0
    fi
    if ! git commit -q -m "${msg}" -- "${known[@]}" >/dev/null 2>"${note}.err"; then
      { echo "NOT PUSHED: git commit failed:"; cat "${note}.err"; } > "${note}"
      rm -f "${note}.err"
      exit 0
    fi
    rm -f "${note}.err"
    local try ok=0 out
    for try in 1 2 3; do
      out="$(git pull -q --rebase --autostash "${C5_GIT_REMOTE}" "${C5_GIT_BRANCH}" 2>&1)" || {
        git rebase --abort >/dev/null 2>&1
        echo "[c5-git] try ${try}: pull --rebase failed: ${out}"
        sleep $((try * 5))
        continue
      }
      if out="$(git push -q "${C5_GIT_REMOTE}" "HEAD:${C5_GIT_BRANCH}" 2>&1)"; then
        ok=1
        break
      fi
      echo "[c5-git] try ${try}: push failed: ${out}"
      sleep $((try * 5))
    done
    {
      if [[ "${ok}" -eq 1 ]]; then
        echo "pushed $(git rev-parse --short HEAD) to ${C5_GIT_REMOTE}/${C5_GIT_BRANCH}: ${msg}"
      else
        echo "NOT PUSHED after 3 tries (committed locally as $(git rev-parse --short HEAD)): ${msg}"
        echo "last error: ${out}"
      fi
      if [[ "${#skipped[@]}" -gt 0 ]]; then
        echo "not added:"
        printf '  %s\n' "${skipped[@]}"
      fi
    } > "${note}"
    cat "${note}"
    exit 0
  )
  # The push note goes into the report (the report itself was already committed; the note is
  # committed with the next step).
  if [[ -f "${note}" && -f "${report}" ]]; then
    {
      echo
      echo "## Push"
      echo
      echo '```'
      cat "${note}"
      echo '```'
    } >> "${report}"
  fi
  return 0
}

# c5_check_python <report.md> <title> [vllm]
#   Before a step: the chosen Python must import the CPU set (and vLLM for the GPU steps). On
#   failure it writes the step's report with what is missing and how to point C5 at the right
#   environment, and returns 1.
c5_check_python() {
  local report="$1" title="$2" mods="numpy, pandas, pyarrow, transformers, yaml"
  [[ "${3:-}" == "vllm" ]] && mods="${mods}, vllm"
  local out
  if out="$("${PYTHON}" -c "import sys, ${mods}; print(sys.executable, sys.version.split()[0])" 2>&1)"; then
    echo "[c5] python ${out}"
    "${PYTHON}" -c "import datasets" >/dev/null 2>&1 || echo "[c5] note: this Python has no 'datasets' package; MuSiQue aliases are skipped and NovelHopQA gold answers come from RAPTOR's copy"
    return 0
  fi
  {
    echo "# C5 end-to-end QA - ${title}: FAILED (Python environment)"
    echo
    echo "Written $(date '+%F %T') on $(hostname). Nothing was run."
    echo
    echo "The Python C5 picked, \`${PYTHON}\`, cannot import: ${mods}."
    echo
    echo '```'
    echo "${out}" | tail -n 5
    echo '```'
    echo
    echo "Point C5 at the study environment (on gpu03 C1 and C2 used /mnt/data/zeinab/raptor/.venv/bin/python3), e.g."
    echo
    echo '```bash'
    echo "C5_PYTHON=/mnt/data/zeinab/raptor/.venv/bin/python3 bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh"
    echo '```'
  } > "${report}"
  echo "FAILED: Python ${PYTHON} lacks one of: ${mods}" > "${report%.md}.summary.txt"
  return 1
}
