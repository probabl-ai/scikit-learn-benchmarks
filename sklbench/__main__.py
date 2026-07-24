# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import sys

from sklbench.config import load_cases_from_script
from sklbench.orchestrator import get_orchestrator_parser, get_parser_description
from sklbench.orchestrator import orchestrate_benchmarks


def main():
    parser = get_orchestrator_parser()
    args = parser.parse_args()
    if args.describe_parser:
        print(get_parser_description(parser))
        return 0

    if args.config is None:
        parser.error("--config is required unless --describe-parser is used")

    bench_cases = [
        case for config in args.config for case in load_cases_from_script(config)
    ]
    return orchestrate_benchmarks(bench_cases, args)


if __name__ == "__main__":
    sys.exit(main())
