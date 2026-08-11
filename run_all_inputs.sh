#!/usr/bin/env bash

set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv/bin/python"
input_root="$project_dir/inputs"

if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

if [[ ! -d "$input_root" ]]; then
    echo "Input directory not found: $input_root" >&2
    exit 1
fi

cd "$project_dir" || exit 1
shopt -s nullglob

case_dirs=("$input_root"/*/)
if (( ${#case_dirs[@]} == 0 )); then
    echo "No input cases found in: $input_root"
    exit 0
fi

success_count=0
skipped_count=0
failed_cases=()

for case_dir in "${case_dirs[@]}"; do
    case_dir="${case_dir%/}"
    case_name="${case_dir##*/}"
    video_path="$case_dir/$case_name.mp4"
    target_dir="$case_dir/target"

    if [[ ! -f "$video_path" ]]; then
        echo "Skip $case_name: missing $video_path"
        ((skipped_count += 1))
        continue
    fi

    has_target=false
    if [[ -d "$target_dir" ]]; then
        for target_file in "$target_dir"/*; do
            if [[ -f "$target_file" ]]; then
                has_target=true
                break
            fi
        done
    fi

    if [[ "$has_target" != true ]]; then
        echo "Skip $case_name: target directory is missing or empty"
        ((skipped_count += 1))
        continue
    fi

    echo
    echo "Running: $case_name"
    if "$python_bin" main.py "name=$case_name"; then
        echo "Completed: $case_name"
        ((success_count += 1))
    else
        echo "Failed: $case_name" >&2
        failed_cases+=("$case_name")
    fi
done

echo
echo "Finished: $success_count succeeded, $skipped_count skipped, ${#failed_cases[@]} failed"

if (( ${#failed_cases[@]} > 0 )); then
    echo "Failed cases: ${failed_cases[*]}" >&2
    exit 1
fi
