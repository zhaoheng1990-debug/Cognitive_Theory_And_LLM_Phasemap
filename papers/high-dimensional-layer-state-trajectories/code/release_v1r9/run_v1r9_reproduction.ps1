param(
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath "$repoRoot\source_data" -PathType Container)) {
    $repoRoot = Split-Path -Parent $repoRoot
}
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = "$repoRoot\source_data\source_data\v1r9_final"
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Source Data directory was not found: $DataRoot"
}

& python "$PSScriptRoot\verify_v1r4_source_data.py" --data-root $DataRoot --output "$repoRoot\verification\v1r9_inherited_endpoint_verification.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "$PSScriptRoot\verify_v1r9_scale_confirmation.py" --data-root $DataRoot --output "$repoRoot\verification\v1r9_scale_confirmation_verification.json"
exit $LASTEXITCODE
