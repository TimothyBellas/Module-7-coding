param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_./:-]*$')]
    [string]$NewModel
)

$ErrorActionPreference = "Stop"
$TranscriptStarted = $false
Push-Location $PSScriptRoot

function Invoke-Compose {
    param([string[]]$ComposeArgs)
    & docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed: $($ComposeArgs -join ' ')"
    }
}

try {
    Start-Transcript -Path (Join-Path $PSScriptRoot "config-verification-results.txt") -Force | Out-Null
    $TranscriptStarted = $true

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Open/install Docker Desktop, then rerun this script."
    }
    & docker info --format '{{.ServerVersion}}'
    if ($LASTEXITCODE -ne 0) {
        throw "Open Docker Desktop and wait for its engine, then rerun this script."
    }

    if (-not (Test-Path -LiteralPath ".env")) {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
    }
    Invoke-Compose -ComposeArgs @("config", "--quiet")
    Invoke-Compose -ComposeArgs @("up", "--build", "-d", "--wait", "--wait-timeout", "180")

    # Read the actual published port, including any shell override of BACKEND_PORT.
    $PortOutput = @(& docker compose port backend 8000)
    if ($LASTEXITCODE -ne 0 -or $PortOutput.Count -eq 0) {
        throw "Could not find the backend's published port."
    }
    if ($PortOutput[0] -notmatch ':(\d+)$') {
        throw "Unexpected backend port output: $($PortOutput[0])"
    }
    $BaseUrl = "http://localhost:$($Matches[1])"
    $Before = Invoke-RestMethod -Uri "$BaseUrl/" -TimeoutSec 15
    $CountBefore = (Invoke-RestMethod -Uri "$BaseUrl/stats" -TimeoutSec 15).documents
    Write-Host "Model before change: $($Before.model)"

    if (-not $NewModel) {
        $NewModel = if ($Before.model -eq "llama3.2:3b") { "llama3.2:1b" } else { "llama3.2:3b" }
    }
    if ($NewModel -eq $Before.model) {
        throw "Choose a different model with -NewModel; the API already uses $NewModel."
    }

    $EnvPath = Join-Path $PSScriptRoot ".env"
    $Original = [System.IO.File]::ReadAllText($EnvPath)
    $Pattern = '(?m)^[ \t]*MODEL_NAME[ \t]*=.*$'
    if ([regex]::IsMatch($Original, $Pattern)) {
        $Updated = [regex]::Replace($Original, $Pattern, "MODEL_NAME=$NewModel")
    } else {
        $Updated = $Original.TrimEnd() + "`r`nMODEL_NAME=$NewModel`r`n"
    }
    [System.IO.File]::WriteAllText($EnvPath, $Updated, [System.Text.UTF8Encoding]::new($false))
    Write-Host "Changed MODEL_NAME in .env to $NewModel"

    # Recreate the backend so Docker reloads env_file. A plain restart keeps old env values.
    Invoke-Compose -ComposeArgs @("up", "-d", "--force-recreate", "--no-deps", "--wait", "--wait-timeout", "180", "backend")
    $After = Invoke-RestMethod -Uri "$BaseUrl/" -TimeoutSec 15
    $Stats = Invoke-RestMethod -Uri "$BaseUrl/stats" -TimeoutSec 15
    $Health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 15
    Write-Host ($After | ConvertTo-Json)
    Write-Host ($Stats | ConvertTo-Json)

    if ($After.model -ne $NewModel -or $Stats.model -ne $NewModel) {
        throw "The running API did not load MODEL_NAME=$NewModel from .env."
    }
    if ($Stats.documents -ne $CountBefore) {
        throw "Stored document count changed during backend recreation."
    }
    if ($Health.ollama -ne "connected" -or $Health.chromadb -ne "accessible") {
        throw "The backend cannot reach its dependencies after recreation."
    }

    Write-Host "PASS: MODEL_NAME changed from $($Before.model) to $($After.model)."
    Write-Host "PASS: ChromaDB count is unchanged and Ollama is connected."
    Write-Host "Evidence: config-verification-results.txt"
    Write-Host "The new model remains in .env and the containers remain running."
    Write-Host "Before using /ask, download it with: docker compose exec ollama ollama pull $NewModel"
}
finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    Pop-Location
}
