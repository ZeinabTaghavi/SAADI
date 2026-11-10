#!/usr/bin/env bash
# Shared shell helpers of C8. Sourced after env.sh by every sh/*.sh.
#
#   c8_run_step <step_id> <title> <report.md> <log> <command...>
#       runs the command with its output in the log; when it dies before writing its report,
#       writes a report with the exit code and the log tail. Returns the exit code.
#   c8_push <dataset|all> <step> <report.md>
#       commits ONLY results/rebuttal/c8_long_context/ and experiments/rebuttal/c8_long_context/
#       and pushes to ${C8_GIT_BRANCH}. Never "git add -A". Skips files over 5 MB, .env files and
#       anything that looks like a key or token. "git pull --rebase" and up to 3 tries; a failed
#       push is written into the report and never stops the experiment (always returns 0).
#   c8_check_python <report.md> <title>
#       the CPU Python must import what C8 needs; otherwise writes the report and returns 1.

C8_GIT_PATHS=("results/rebuttal/c8_long_context" "experiments/rebuttal/c8_long_context")

c8_run_step() {
  local step="$1" title="$2" report="$3" log="$4"
  shift 4
  [[ -f "${report}" ]] && mv -f "${report}" "${report%.md}.prev.md"
  rm -f "${report%.md}.json"
  echo "[c8] $(date '+%F %T') start ${step}: $*" >> "${log}"
  "$@" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  if [[ ! -f "${report}" ]]; then
    {
      echo "# C8 long-context reader - ${title}: FAILED before writing its report"
      echo
      echo "Command: \`$*\`"
      echo
      echo "Exit code: ${status}. Last 80 log lines (${log#"${C8_ROOT_DIR}/"}):"
      echo
      echo '```'
      tail -n 80 "${log}" || true
      echo '```'
    } > "${report}"
    echo "FAILED before writing its report (exit ${status})" > "${report%.md}.summary.txt"
    [[ "${status}" -eq 0 ]] && status=1
  fi
  echo "[c8] $(date '+%F %T') end ${step}: exit ${status}" >> "${log}"
  return "${status}"
}

# One-line result for the commit message: the first line of <report>.summary.txt, else the
# report's first heading.
c8_summary_line() {
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

# Secrets: key and token shapes (Hugging Face, OpenAI/Anthropic, GitHub, AWS, private keys,
# explicit api_key assignments). Same list as C5.
C8_SECRET_RE='(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api[_-]?key|access[_-]?token|secret)["'"'"' ]*[:=][ ]*["'"'"'][A-Za-z0-9_-]{16,})'

# C5's GPU scripts push from the same checkout: C8 takes the SAME lock, so no two git calls overlap.
C8_LOCK_DIR_NAME="c5_push.lock.d"

c8_lock_acquire() {
  local lockdir="${C8_ROOT_DIR}/.git/${C8_LOCK_DIR_NAME}" waited=0
  while ! mkdir "${lockdir}" 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
    if [[ "${waited}" -ge 600 ]]; then
      echo "[c8-git] lock held for 10 min; removing a stale lock ${lockdir}"
      rm -rf "${lockdir}"
    fi
  done
  echo "$$" > "${lockdir}/pid"
}

c8_lock_release() {
  rm -rf "${C8_ROOT_DIR}/.git/${C8_LOCK_DIR_NAME}"
}

c8_push() {
  local dataset="$1" step="$2" report="$3"
  local note="${report%.md}.push.txt"
  local msg="C8 ${dataset} ${step}: $(c8_summary_line "${report}")"
  if [[ "${C8_GIT_PUSH}" != "1" ]]; then
    echo "push disabled (C8_GIT_PUSH=${C8_GIT_PUSH})" > "${note}"
    return 0
  fi
  (
    cd "${C8_ROOT_DIR}" || exit 0
    c8_lock_acquire
    trap c8_lock_release EXIT
    local branch
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    if [[ "${branch}" != "${C8_GIT_BRANCH}" ]]; then
      echo "NOT PUSHED: checkout is on branch '${branch}', expected '${C8_GIT_BRANCH}'." | tee "${note}"
      exit 0
    fi
    local max_bytes=$((C8_GIT_MAX_FILE_MB * 1024 * 1024)) f size skipped=()
    local add=()
    while IFS= read -r -d '' f; do
      case "${f}" in
        */.env|*/.env.*|*.env|*/__pycache__/*|*.pyc|*.tmp) skipped+=("${f} (excluded type)"); continue ;;
        *.safetensors|*.bin|*.pt|*.pth|*.gguf) skipped+=("${f} (model weights)"); continue ;;
        *" "[0-9].*) skipped+=("${f} (looks like a file-sync conflict copy)"); continue ;;
      esac
      size=$(wc -c < "${f}" 2>/dev/null || echo 0)
      if [[ "${size}" -gt "${max_bytes}" ]]; then
        skipped+=("${f} (${size} bytes > ${C8_GIT_MAX_FILE_MB} MB)")
        continue
      fi
      if grep -Eq "${C8_SECRET_RE}" "${f}" 2>/dev/null; then
        skipped+=("${f} (looks like it holds a key or token)")
        continue
      fi
      add+=("${f}")
    done < <(find "${C8_GIT_PATHS[@]}" -type f -print0 2>/dev/null)
    if [[ "${#add[@]}" -eq 0 ]]; then
      echo "nothing to commit" > "${note}"
      exit 0
    fi
    git add -f -- "${add[@]}" >/dev/null 2>&1
    # Commit only the C8 paths, whatever else is staged in this checkout.
    local known=() d
    for d in "${C8_GIT_PATHS[@]}"; do
      [[ -n "$(git ls-files -- "${d}" | head -n 1)" ]] && known+=("${d}")
    done
    if [[ "${#known[@]}" -eq 0 ]] || git diff --cached --quiet -- "${known[@]}"; then
      echo "no changes in the C8 folders since the last commit" > "${note}"
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
      out="$(git pull -q --rebase --autostash "${C8_GIT_REMOTE}" "${C8_GIT_BRANCH}" 2>&1)" || {
        git rebase --abort >/dev/null 2>&1
        echo "[c8-git] try ${try}: pull --rebase failed: ${out}"
        sleep $((try * 5))
        continue
      }
      if out="$(git push -q "${C8_GIT_REMOTE}" "HEAD:${C8_GIT_BRANCH}" 2>&1)"; then
        ok=1
        break
      fi
      echo "[c8-git] try ${try}: push failed: ${out}"
      sleep $((try * 5))
    done
    {
      if [[ "${ok}" -eq 1 ]]; then
        echo "pushed $(git rev-parse --short HEAD) to ${C8_GIT_REMOTE}/${C8_GIT_BRANCH}: ${msg}"
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
  # The push note goes into the report (committed with the next step).
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

c8_check_python() {
  local report="$1" title="$2" mods="numpy, pandas, transformers, yaml"
  local out
  if out="$("${PYTHON}" -c "import sys, ${mods}; print(sys.executable, sys.version.split()[0])" 2>&1)"; then
    echo "[c8] python ${out}"
    return 0
  fi
  {
    echo "# C8 long-context reader - ${title}: FAILED (Python environment)"
    echo
    echo "Written $(date '+%F %T') on $(hostname). Nothing was run."
    echo
    echo "The Python C8 picked (C5's choice), \`${PYTHON}\`, cannot import: ${mods}."
    echo
    echo '```'
    echo "${out}" | tail -n 5
    echo '```'
    echo
    echo "Point C8 at the study environment, e.g."
    echo
    echo '```bash'
    echo "C5_PYTHON=/mnt/data/zeinab/raptor/.venv/bin/python3 bash experiments/rebuttal/c8_long_context/sh/00_prepare.sh"
    echo '```'
  } > "${report}"
  echo "FAILED: Python ${PYTHON} lacks one of: ${mods}" > "${report%.md}.summary.txt"
  return 1
}
