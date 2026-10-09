param([switch]$TestRag)

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

function Get-ProjectPort {
    param([string]$Name, [string]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $value) {
        foreach ($line in Get-Content -LiteralPath ".env") {
            if ($line -match "^\s*$Name\s*=\s*(\d+)\s*$") {
                $value = $Matches[1]
                break
            }
        }
    }
    if (-not $value) { $value = $Default }
    if ($value -notmatch '^\d+$' -or [int]$value -lt 1 -or [int]$value -gt 65535) {
        throw "Invalid $Name port: $value"
    }
    return $value
}

function Assert-Healthy {
    param([string]$BaseUrl)
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 15
    Write-Host ($health | ConvertTo-Json)
    if ($health.status -ne "healthy" -or $health.ollama -ne "connected" -or
        $health.chromadb -ne "accessible" -or $health.ollama_url -ne "http://ollama:11434") {
        throw "Backend cannot reach its dependencies correctly."
    }
    return $health
}

try {
    Start-Transcript -Path (Join-Path $PSScriptRoot "verification-results.txt") -Force | Out-Null
    $TranscriptStarted = $true

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found. Install/open Docker Desktop, then rerun this script."
    }
    & docker info --format '{{.ServerVersion}}'
    if ($LASTEXITCODE -ne 0) {
        throw "Open Docker Desktop and wait until the engine is running, then rerun this script."
    }

    $BackendPort = Get-ProjectPort -Name "BACKEND_PORT" -Default "8000"
    $OllamaPort = Get-ProjectPort -Name "OLLAMA_PORT" -Default "11434"
    $BaseUrl = "http://localhost:$BackendPort"
    $OllamaBaseUrl = "http://localhost:$OllamaPort"

    Write-Host "Checking Compose configuration..."
    Invoke-Compose -ComposeArgs @("config", "--quiet")
    Write-Host "Building and starting both containers..."
    Invoke-Compose -ComposeArgs @("up", "--build", "-d", "--wait", "--wait-timeout", "180")

    Write-Host "Checking the root endpoint..."
    $root = Invoke-RestMethod -Uri "$BaseUrl/" -TimeoutSec 15
    Write-Host ($root | ConvertTo-Json)
    if ($root.message -ne "RAG API running in Docker") {
        throw "Unexpected root endpoint response."
    }

    Write-Host "Checking backend-to-Ollama connectivity..."
    $null = Assert-Healthy -BaseUrl $BaseUrl
    Write-Host "Ingesting documents (first run may download Chroma's embedding model)..."
    $ingest = Invoke-RestMethod -Uri "$BaseUrl/ingest" -Method Post -TimeoutSec 600
    Write-Host ($ingest | ConvertTo-Json)
    $Before = (Invoke-RestMethod -Uri "$BaseUrl/stats" -TimeoutSec 15).documents
    if ($Before -le 0) { throw "No documents were stored; persistence cannot be verified." }
    Write-Host "Document count before restart: $Before"

    if ($TestRag) {
        Write-Host "Downloading the generation model into Ollama's named volume..."
        Invoke-Compose -ComposeArgs @("exec", "-T", "ollama", "ollama", "pull", $root.model)
        $body = @{ question = "What is Docker Compose?" } | ConvertTo-Json
        $answer = Invoke-RestMethod -Uri "$BaseUrl/ask" -Method Post `
            -ContentType "application/json" -Body $body -TimeoutSec 240
        Write-Host ($answer | ConvertTo-Json -Depth 6)
        if (-not $answer.answer -or $answer.chunks_retrieved -le 0 -or $answer.sources.Count -le 0) {
            throw "The RAG response did not include retrieved sources."
        }
    }

    $ModelsBefore = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -TimeoutSec 15
    $DigestsBefore = (@($ModelsBefore.models | ForEach-Object { $_.digest } | Sort-Object) -join ",")

    Write-Host "Stopping/removing containers while preserving the named volumes..."
    Invoke-Compose -ComposeArgs @("down")
    Write-Host "Restarting without ingesting documents again..."
    Invoke-Compose -ComposeArgs @("up", "-d", "--wait", "--wait-timeout", "180")
    $null = Assert-Healthy -BaseUrl $BaseUrl

    $After = (Invoke-RestMethod -Uri "$BaseUrl/stats" -TimeoutSec 15).documents
    Write-Host "Document count after restart: $After"
    if ($Before -ne $After) { throw "Persistence failed: $Before became $After." }

    $ModelsAfter = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -TimeoutSec 15
    $DigestsAfter = (@($ModelsAfter.models | ForEach-Object { $_.digest } | Sort-Object) -join ",")
    if ($DigestsBefore -ne $DigestsAfter) { throw "Ollama's model list changed after restart." }

    Write-Host "PASS: root response, Ollama connection, and nonzero ChromaDB count survived down/up."
    if (@($ModelsBefore.models).Count -gt 0) {
        Write-Host "PASS: Ollama model digests also survived down/up."
    } else {
        Write-Host "No generation model was downloaded. Use -TestRag to check generation and model persistence."
    }
    Write-Host "Swagger UI: $BaseUrl/docs"
    Write-Host "Evidence saved in verification-results.txt. Containers are still running."
}
finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    Pop-Location
}
