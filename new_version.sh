#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C
export GIT_OPTIONAL_LOCKS=0

usage() {
    cat <<EOF
Usage: ${0##*/} VERSION [--dry-run] [--commit]

Copy the highest numeric EtaZero_V* version and register the new version as 开发中.
Only tracked source files are copied; build/, runs/, data/, tmp/, caches and
*.cfg.local are excluded. Creation requires a clean Git worktree.

  --dry-run  Preview the operation and any dirty-worktree blocker without writing.
  --commit   Create one commit containing the new directory and AGENTS.md.

Examples:
  bash ${0##*/} 1 --dry-run
  bash ${0##*/} 1
  bash ${0##*/} 1 --commit
EOF
}

die() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

# Compare decimal components as strings, avoiding octal and integer overflow.
compare_versions() {
    local -a left_parts right_parts
    local index count left right
    IFS='.' read -r -a left_parts <<< "$1"
    IFS='.' read -r -a right_parts <<< "$2"
    count=${#left_parts[@]}
    (( ${#right_parts[@]} <= count )) || count=${#right_parts[@]}
    for ((index = 0; index < count; index++)); do
        left=${left_parts[index]:-0}
        right=${right_parts[index]:-0}
        while [[ ${#left} -gt 1 && $left == 0* ]]; do left=${left:1}; done
        while [[ ${#right} -gt 1 && $right == 0* ]]; do right=${right:1}; done
        if (( ${#left} > ${#right} )); then printf '1\n'; return; fi
        if (( ${#left} < ${#right} )); then printf '%s\n' -1; return; fi
        if [[ $left > $right ]]; then printf '1\n'; return; fi
        if [[ $left < $right ]]; then printf '%s\n' -1; return; fi
    done
    printf '0\n'
}

excluded_path() {
    case "$1" in
        build|build/*|runs|runs/*|data|data/*|tmp|tmp/*|\
        __pycache__/*|*/__pycache__/*|.pytest_cache/*|*/.pytest_cache/*|*.cfg.local)
            return 0 ;;
        *) return 1 ;;
    esac
}

target_version=''
dry_run=0
commit=0
for argument in "$@"; do
    case "$argument" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1 ;;
        --commit) commit=1 ;;
        --*) die "unknown option '$argument'" ;;
        *) [[ -z $target_version ]] || die 'specify exactly one target version'
           target_version=$argument ;;
    esac
done
[[ -n $target_version ]] || { usage >&2; exit 2; }
[[ $target_version =~ ^[0-9]+([.][0-9]+)*$ ]] || die "invalid version '$target_version'; use a numeric version such as 0.10, 1.0, or 2"

command -v git >/dev/null 2>&1 || die 'git is required'
command -v perl >/dev/null 2>&1 || die 'perl is required'
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "$script_root" rev-parse --show-toplevel 2>/dev/null) || die 'script is not inside a Git repository'
cd -- "$repo_root"
[[ $(pwd -P) == "$script_root" ]] || die 'place this script in the Git repository root'
[[ -f AGENTS.md && ! -L AGENTS.md ]] || die 'AGENTS.md must be a regular file in the repository root'
source_head=$(git rev-parse HEAD) || die 'repository must have a committed baseline'

source_version=''
versions=()
shopt -s nullglob
for directory in EtaZero_V*; do
    version=${directory#EtaZero_V}
    [[ $version =~ ^[0-9]+([.][0-9]+)*$ && -d $directory ]] || continue
    [[ ! -L $directory ]] || die "version directory '$directory' must not be a symlink"
    for existing_version in "${versions[@]}"; do
        [[ $(compare_versions "$version" "$existing_version") != 0 ]] || \
            die "equivalent version directories: EtaZero_V$existing_version and $directory"
    done
    versions+=("$version")
    if [[ -z $source_version ]] || [[ $(compare_versions "$version" "$source_version") == 1 ]]; then
        source_version=$version
    fi
done
[[ -n $source_version ]] || die 'no numeric EtaZero_V* directory found'
source_dir="EtaZero_V$source_version"
target_dir="EtaZero_V$target_version"
[[ ! -e $target_dir && ! -L $target_dir ]] || die "target '$target_dir' already exists"
[[ $(compare_versions "$target_version" "$source_version") == 1 ]] || \
    die "target version $target_version must be greater than current version $source_version"

# Validate only the version-status section and append a row without rewriting
# existing states, routing instructions, or references to other projects.
updated_agents=$(SOURCE_DIR=$source_dir TARGET_DIR=$target_dir perl - AGENTS.md <<'PERL'
use strict;
use warnings;
use utf8;
use open qw(:std :encoding(UTF-8));
local $/;
my $text = <>;
my @lines = split /\n/, $text, -1;
my @sections = grep { $lines[$_] =~ /^## 版本状态[ \t]*$/ } 0 .. $#lines;
@sections == 1 or die "Error: AGENTS.md must contain one '版本状态' section\n";
my $start = $sections[0] + 1;
my $end = $start;
$end++ while $end <= $#lines && $lines[$end] !~ /^## /;
my @headers = grep { $lines[$_] =~ /^\|[ \t]*版本目录[ \t]*\|[ \t]*状态[ \t]*\|[ \t]*$/ } $start .. $end - 1;
@headers == 1 or die "Error: version-status table is missing or ambiguous\n";
my $row = $headers[0] + 1;
$row < $end && $lines[$row] =~ /^\|[ \t]*:?-{3,}:?[ \t]*\|[ \t]*:?-{3,}:?[ \t]*\|[ \t]*$/
    or die "Error: invalid version-status table separator\n";
sub canonical {
    my @parts = split /\./, shift;
    s/^0+(?=[0-9])// for @parts;
    pop @parts while @parts > 1 && $parts[-1] eq '0';
    return join '.', @parts;
}
my %seen;
my $source_found = 0;
my $target = canonical(substr $ENV{TARGET_DIR}, length 'EtaZero_V');
$row++;
while ($row < $end && $lines[$row] =~ /^\|/) {
    my ($directory, $version) = $lines[$row] =~ /^\|[ \t]*(EtaZero_V([0-9]+(?:\.[0-9]+)*))[ \t]*\|[ \t]*(?:开发中|已冻结)[ \t]*\|[ \t]*$/;
    defined $directory or die "Error: invalid version-status row: $lines[$row]\n";
    my $key = canonical($version);
    !$seen{$key}++ or die "Error: duplicate version-status entry: $directory\n";
    $key ne $target or die "Error: target version already registered: $directory\n";
    $source_found = 1 if $directory eq $ENV{SOURCE_DIR};
    $row++;
}
$source_found or die "Error: source $ENV{SOURCE_DIR} is not registered in the version-status table\n";
splice @lines, $row, 0, "| $ENV{TARGET_DIR} | 开发中 |";
print join "\n", @lines;
PERL
)

mapfile -d '' -t tracked_files < <(git ls-files -z -- "$source_dir/")
listing_pid=$!
wait "$listing_pid" || die 'could not list tracked source files'
copy_files=()
excluded_files=0
for source_path in "${tracked_files[@]}"; do
    relative_path=${source_path#"$source_dir/"}
    if excluded_path "$relative_path"; then
        ((excluded_files += 1))
    else
        copy_files+=("$source_path")
    fi
done
(( ${#copy_files[@]} > 0 )) || die "source '$source_dir' has no tracked source files"
git ls-files --error-unmatch -- "$source_dir/README.md" >/dev/null 2>&1 || die 'source README.md must be tracked'

status=$(git status --porcelain=v1 --untracked-files=all)
printf 'Source: %s\nTarget: %s\n' "$source_dir" "$target_dir"
printf 'Tracked files to copy: %d; excluded artifacts/local files: %d\n' "${#copy_files[@]}" "$excluded_files"
printf 'AGENTS.md: append | %s | 开发中 |; preserve existing states\n' "$target_dir"
printf 'Text updates: README title, Markdown usage examples and CLI descriptions\n'
if (( commit )); then printf 'Git: one commit for %s and AGENTS.md\n' "$target_dir";
else printf 'Git: leave generated changes for review\n'; fi

if (( dry_run )); then
    printf 'Dry run: no files or commits created.\n'
    if [[ -n $status ]]; then
        printf 'Creation is blocked by uncommitted changes:\n%s\n' "$status"
    fi
    exit 0
fi
[[ -z $status ]] || die "repository has uncommitted changes; commit the intended baseline first:
$status"
staging_root=$(mktemp -d "$repo_root/.new_version.XXXXXX")
staging_dir="$staging_root/$target_dir"
target_created=0
completed=0
cleanup() {
    if (( target_created && ! completed )); then rm -rf -- "$repo_root/$target_dir"; fi
    rm -rf -- "$staging_root"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -- "$staging_dir"
for source_path in "${copy_files[@]}"; do
    [[ -f $source_path || -L $source_path ]] || die "tracked source '$source_path' is missing or is not a file"
    relative_path=${source_path#"$source_dir/"}
    destination_path="$staging_dir/$relative_path"
    mkdir -p -- "${destination_path%/*}"
    cp -a -- "$source_path" "$destination_path"
done

# Restrict automatic path edits to fenced examples and the explicit 'in this
# directory' note. Historical prose and Markdown links remain for manual review.
updated_documents=0
for source_path in "${copy_files[@]}"; do
    relative_path=${source_path#"$source_dir/"}
    [[ $relative_path == *.md ]] || continue
    document="$staging_dir/$relative_path"
    [[ ! -L $document ]] || continue
    OLD_VERSION=$source_version NEW_VERSION=$target_version DOCUMENT=$relative_path \
        perl -i -pe '
        BEGIN { $old = $ENV{OLD_VERSION}; $new = $ENV{NEW_VERSION}; }
        if ($ENV{DOCUMENT} eq "README.md") {
            s/^(# EtaZero V)\Q$old\E(?=\s|$)/$1$new/;
        }
        if (/^\s*(`{3,}|~{3,})/) {
            $mark = substr($1, 0, 1); $length = length($1);
            if (!$fence) { $fence = $mark; $fence_length = $length; }
            elsif ($mark eq $fence && $length >= $fence_length) { $fence = ""; }
        }
        if ($fence || /^# 在 \QEtaZero_V$old\/\E 下/) {
            s/(?<![A-Za-z0-9_])\QEtaZero_V$old\/\E/EtaZero_V$new\//g;
        }
        ' "$document"
    if ! cmp -s -- "$source_path" "$document"; then ((updated_documents += 1)); fi
done
for relative_path in python/etazero/__main__.py python/etazero/config.py; do
    document="$staging_dir/$relative_path"
    [[ -f $document && ! -L $document ]] || continue
    OLD_VERSION=$source_version perl -i -pe '
        s/\QEtaZero V$ENV{OLD_VERSION} training and evaluation\E/EtaZero training and evaluation/g;
        s/\QSelected algorithm\/search combination is not implemented in V$ENV{OLD_VERSION}\E(?=["\x27])/Selected algorithm\/search combination is not implemented in this version/g;
    ' "$document"
done

cp -a -- AGENTS.md "$staging_root/AGENTS.md"
printf '%s\n' "$updated_agents" > "$staging_root/AGENTS.md"

printf 'Updated Markdown documents: %d\n' "$updated_documents"
printf 'Remaining source-version references (review historical links before changing them):\n'
version_pattern=${source_version//./\\.}
reference_pattern="(^|[^[:alnum:]_])(EtaZero_V${version_pattern}/|(EtaZero V|in V)${version_pattern}([^0-9.]|\\.[^0-9]|\\.$|$))"
if references=$(grep -rnIE -- "$reference_pattern" "$staging_dir"); then
    printf '%s\n' "${references//"$staging_dir"/"$target_dir"}"
    printf 'The references above were preserved.\n'
else
    grep_status=$?
    (( grep_status == 1 )) || die 'could not scan source-version references'
    printf 'None.\n'
fi

# Refuse concurrent baseline changes before publishing the staged copy.
[[ $(git rev-parse HEAD) == "$source_head" ]] || die 'Git HEAD changed during creation; retry from the intended baseline'
status=$(git status --porcelain=v1 --untracked-files=all -- . ":(exclude)${staging_root##*/}/**")
[[ -z $status ]] || die "worktree changed during creation:
$status"

# Reserve the target exclusively. Keep the short directory/table publication
# together; on failure before the table update, cleanup removes only our target.
trap '' HUP INT TERM
mkdir -- "$target_dir"
target_created=1
mv -T -- "$staging_dir" "$target_dir"
mv -T -- "$staging_root/AGENTS.md" AGENTS.md
completed=1
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Created %s from %s\n' "$target_dir" "$source_dir"
if (( commit )); then
    if ! git add -- AGENTS.md "$target_dir" || \
       ! git commit -m "Create EtaZero V${target_version} from V${source_version}"; then
        die 'automatic commit failed; generated files and AGENTS.md are preserved for review/retry'
    fi
fi
