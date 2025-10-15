#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http//:www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

$env:OVMS_DIR=$PSScriptRoot
$env:VENV_DIR=Join-Path $env:OVMS_DIR "..\.venv"
$activateScript=Join-Path $env:VENV_DIR "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated"
} else {
    Write-Host "Warning: Virtual environment not found at $env:VENV_DIR" -ForegroundColor Yellow
}

$env:PATH="$env:OVMS_DIR;$env:PATH"
Write-Host "OpenVINO Model Server Environment Initialized"
