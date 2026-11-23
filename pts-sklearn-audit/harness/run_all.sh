#!/bin/bash
# Install pts/scikit-learn-2.0.0 with the Phoronix Test Suite client and run
# its entries one at a time, each bench_*.py process profiled by shim/python3.
# Run inside the pixi env: `pixi run ./run_all.sh [ENTRIES_FILE]`.
set -u
W=$(cd "$(dirname "$0")" && pwd)
ENTRIES=${1:-$W/entries.txt}
RUN_ID=${RUN_ID:-run}

# PTS reads HOME as DEBUG_REAL_HOME: pip --user installs and scikit_learn_data
# land in this private home, not the real one.
export HOME=$W/home
export PTS_SILENT_MODE=1 NO_EXTERNAL_DEPENDENCIES=1
export PIP_UPLOADED_PRIOR_TO=2023-05-10T00:00:00Z
export PROF_DIR=$W/prof/$RUN_ID PROF_SITECUSTOM=$W/sitecustom
export RUN_LIMIT_S=${RUN_LIMIT_S:-2400}
export PATH=$W/shim:$PATH
mkdir -p "$HOME" "$PROF_DIR"
LOG=$W/logs/$RUN_ID
mkdir -p "$LOG"
# With posix_getpwuid, PTS takes the home directory from the password database
# and ignores HOME.
PTS="php -d disable_functions=posix_getpwuid $W/ptsrc/pts-core/phoronix-test-suite.php"

echo "$(date -Is) start RUN_ID=$RUN_ID slot=${BENCHLOCK_SLOT:-none} limit=${RUN_LIMIT_S}s"

yes n | $PTS user-config-set \
    SaveResults=TRUE UploadResults=FALSE OpenBrowser=FALSE \
    PromptForTestIdentifier=FALSE PromptForTestDescription=FALSE PromptSaveName=FALSE \
    RunAllTestCombinations=FALSE Configured=TRUE \
    AlwaysUploadResultsToOpenBenchmarking=FALSE AnonymousUsageReporting=FALSE \
    > "$LOG/config.log" 2>&1
if [ ! -f "$HOME/.phoronix-test-suite/user-config.xml" ]; then
    echo "$(date -Is) PTS did not use HOME=$HOME, stopping"
    exit 1
fi

INSTALLED=$HOME/.phoronix-test-suite/installed-tests/pts/scikit-learn-2.0.0
if [ ! -x "$INSTALLED/scikit-learn" ]; then
    echo "$(date -Is) installing pts/scikit-learn-2.0.0"
    yes n | $PTS batch-install pts/scikit-learn-2.0.0 > "$LOG/install.log" 2>&1
    if [ ! -x "$INSTALLED/scikit-learn" ]; then
        echo "$(date -Is) install failed, see $LOG/install.log"
        exit 1
    fi
fi
cp "$INSTALLED"/install.log "$LOG/pts-install-script.log" 2>/dev/null
"$CONDA_PREFIX/bin/python3" -c '
import sys, numpy, scipy, sklearn, joblib, threadpoolctl, pprint
print(sys.version)
for m in (numpy, scipy, sklearn, joblib, threadpoolctl):
    print(m.__name__, m.__version__, m.__file__)
pprint.pprint(threadpoolctl.threadpool_info())
' > "$LOG/versions.txt" 2>&1
echo "$(date -Is) installed:"; head -6 "$LOG/versions.txt"

# I0 offsets the entry numbers, so that a resumed sweep doesn't reuse PTS result names.
i=${I0:-0}
while IFS='|' read -r value name; do
    i=$((i + 1))
    [ -z "$value" ] && continue
    echo "$(date -Is) [$i] $name"
    TEST_RESULTS_NAME=$(printf 'skl-%s-%02d' "$RUN_ID" "$i") \
    TEST_RESULTS_IDENTIFIER=$RUN_ID \
    TEST_RESULTS_DESCRIPTION="$name" \
    PRESET_OPTIONS_VALUES="scikit-learn.benchmark=$value" \
        $PTS batch-run pts/scikit-learn-2.0.0 < /dev/null \
        > "$LOG/$(printf '%02d' "$i").log" 2>&1
    echo "$(date -Is) [$i] done rc=$?"
done < "$ENTRIES"
echo "$(date -Is) all done"
