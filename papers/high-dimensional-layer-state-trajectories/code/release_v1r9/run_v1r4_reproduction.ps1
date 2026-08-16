param(
    [ValidateSet("source-data", "full-model")]
    [string]$Mode = "source-data",
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# The GitHub release keeps this script one directory below the paper root,
# whereas the standalone archive keeps it directly below the package root.
if (-not (Test-Path -LiteralPath "$repoRoot\source_data" -PathType Container)) {
    $candidateRoot = Split-Path -Parent $repoRoot
    if (Test-Path -LiteralPath "$candidateRoot\source_data" -PathType Container) {
        $repoRoot = $candidateRoot
    }
}

if ($Mode -eq "source-data") {
    if ([string]::IsNullOrWhiteSpace($DataRoot)) {
        $DataRoot = "$repoRoot\source_data\source_data\v1r4_final"
        if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
            $DataRoot = "$repoRoot\source_data"
        }
    }
    if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
        Write-Host "Source Data directory was not found: $DataRoot"
        Write-Host "Pass -DataRoot <path-to-paired-Source-Data> when using the separate Zenodo Code archive."
        exit 2
    }
    & python "$PSScriptRoot\verify_v1r4_source_data.py" `
        --data-root $DataRoot `
        --output "$repoRoot\verification\v1r4_source_data_verification.json"
    exit $LASTEXITCODE
}

Write-Host "Full-model mode is not bundled with model weights or raw hidden-state arrays."
Write-Host "Use the documented experiment runners under code/extension_v0_51 through code/extension_v0_56 after providing licensed checkpoints and the required runtime."
exit 2
