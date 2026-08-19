param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("import", "proposal", "smoke", "paper", "resume")]
    [string]$Action,
    [string]$Archive,
    [string]$Data,
    [string]$RepositoryRoot,
    [string]$RunDir,
    [string]$SourceRunDir,
    [string]$Config,
    [string]$OutputRoot,
    [string[]]$Devices,
    [string[]]$Models,
    [string[]]$FeatureSets,
    [string[]]$Variants,
    [switch]$ApproveDerivedAR,
    [ValidateSet("auto", "cuda", "mps", "cpu")]
    [string]$Accelerator = "cuda",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"

foreach ($Argument in $ForwardArgs) {
    if ($Argument -eq "--profile" -or $Argument -like "--profile=*") {
        [Console]::Error.WriteLine("profile is owned by the launcher action")
        exit 2
    }
}

uv sync --project hqrc_v3 --extra accelerator --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "import") {
    $CliArgs = @()
    if ($Archive) { $CliArgs += @("--archive", $Archive) }
    if ($Data) { $CliArgs += @("--data", $Data) }
    if ($RepositoryRoot) { $CliArgs += @("--repository-root", $RepositoryRoot) }
    if ($RunDir) { $CliArgs += @("--run-dir", $RunDir) }
    $CliArgs += $ForwardArgs
    uv run --project hqrc_v3 --extra accelerator --locked hqrc import-paper-source @CliArgs
} else {
    $Profile = if ($Action -eq "smoke") { "smoke" } else { "paper" }
    $CliArgs = @("--profile", $Profile, "--accelerator", $Accelerator)
    if ($Action -eq "smoke") {
        $CliArgs += @(
            "--draws", "4", "--tune", "4", "--chains", "4", "--cores", "1",
            "--target-accept", "0.9"
        )
    }
    if ($SourceRunDir) { $CliArgs += @("--source-run-dir", $SourceRunDir) }
    if ($Config) { $CliArgs += @("--config", $Config) }
    if ($OutputRoot) { $CliArgs += @("--output-root", $OutputRoot) }
    if ($Devices) { $CliArgs += "--devices"; $CliArgs += $Devices }
    if ($Models) { $CliArgs += "--models"; $CliArgs += $Models }
    if ($FeatureSets) { $CliArgs += "--feature-sets"; $CliArgs += $FeatureSets }
    if ($Variants) { $CliArgs += "--variants"; $CliArgs += $Variants }
    if ($ApproveDerivedAR) { $CliArgs += "--approve-derived-ar" }
    $CliArgs += $ForwardArgs
    uv run --project hqrc_v3 --extra accelerator --locked hqrc run-loeo-accelerated @CliArgs
}
exit $LASTEXITCODE
