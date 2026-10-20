param(
    [string]$Token = $env:VERCEL_TOKEN,
    [string]$Project = "satari-code"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$web = Join-Path $root "web"

if (-not $Token) { throw "Set VERCEL_TOKEN or pass -Token." }
if (-not (Test-Path $web)) { throw "web/ not found at $web" }

Push-Location $web
try {
    Write-Host "Deploying $web to Vercel (production)..."
    npx --yes vercel@latest deploy . --prod --yes --token $Token --name $Project
    if ($LASTEXITCODE -ne 0) { throw "vercel deploy failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
