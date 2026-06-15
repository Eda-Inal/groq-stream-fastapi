#   .\dev.ps1 api
#   .\dev.ps1 tools
param([string]$service)

switch ($service) {
    "api"   { .venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload } #.\dev.ps1 api
    "tools" { .venv\Scripts\uvicorn app.tool_server.main:app --host 0.0.0.0 --port 8001 --reload } #.\dev.ps1 tools
    default { Write-Host "use: .\dev.ps1 api | .\dev.ps1 tools" }
}
