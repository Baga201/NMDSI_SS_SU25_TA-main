
# PowerShell script to automate agent setup, backend, and frontend startup
# Optional: bootstrap agents for a clean environment
# Enable by setting environment variable DELETE_OLD_AGENTS=1
if ($env:DELETE_OLD_AGENTS -eq "1") {
	Write-Host "Deleting old agents and bootstrapping a clean setup…"
	# Propagate WRITE_ENV_ON_CREATE into the CLI or use the explicit flag
	if ($env:WRITE_ENV_ON_CREATE -eq "1") {
		python scripts/run_team.py --delete-old-agents --write-env-updates
	} else {
		python scripts/run_team.py --delete-old-agents
	}
}

# Start FastAPI backend (configurable)
# Avoid reserved automatic variable names: use bindHost/bindPort
$bindHost = if ($env:BACKEND_HOST) { $env:BACKEND_HOST } else { '127.0.0.1' }
$bindPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { '8001' }

# UI host to show to the user (use localhost when binding to 0.0.0.0)
$uiHost = if ($bindHost -eq '0.0.0.0') { 'localhost' } else { $bindHost }

Write-Host ("Starting backend on {0}:{1} (logs: {2})" -f $bindHost, $bindPort, (Join-Path $PSScriptRoot 'backend.log'))

# Prefer uvicorn on PATH, otherwise fallback to `python -m uvicorn`
 $uvPath = (Get-Command uvicorn -ErrorAction SilentlyContinue)?.Source
if ($uvPath) {
	$exe = 'uvicorn'
	$uvArgs = "scripts.backend:app --host $bindHost --port $bindPort --reload"
} else {
	$py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
	if (-not $py) { $py = 'python' }
	$exe = $py
	$uvArgs = "-m uvicorn scripts.backend:app --host $bindHost --port $bindPort --reload"
}

# Start process and redirect output to log files so users can inspect startup errors
$outLog = Join-Path $PSScriptRoot 'backend.log'
$errLog = Join-Path $PSScriptRoot 'backend.err.log'
try {
	$proc = Start-Process -FilePath $exe -ArgumentList $uvArgs -WorkingDirectory $PSScriptRoot -NoNewWindow -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
		# Persist PID so we can stop the backend later
		try {
			$pidFile = Join-Path $PSScriptRoot 'backend.pid'
			Set-Content -Path $pidFile -Value $proc.Id -Encoding ASCII
		} catch {
			Write-Host "Warning: failed to write PID file: $_"
		}

		# Also persist into .env as BACKEND_PID for canonical storage (backup .env first)
		try {
			$envPath = Join-Path $PSScriptRoot '.env'
			if (Test-Path $envPath) {
				Copy-Item $envPath ($envPath + '.bak') -Force -ErrorAction SilentlyContinue
				$lines = Get-Content $envPath -ErrorAction SilentlyContinue
			} else {
				$lines = @()
			}
			$found = $false
			$newLines = @()
			foreach ($l in $lines) {
				if ($l -match '^[ \t]*BACKEND_PID\s*=') {
					$newLines += ("BACKEND_PID=$($proc.Id)")
					$found = $true
				} else {
					$newLines += $l
				}
			}
			if (-not $found) { $newLines += ("BACKEND_PID=$($proc.Id)") }
			$newLines | Set-Content -Path $envPath -Encoding UTF8
		} catch {
			Write-Host "Warning: failed to write BACKEND_PID to .env: $_"
		}
} catch {
	Write-Host "Failed to start backend process: $_"
	Write-Host "Try running: $exe $uvArgs"
}

# Wait for backend to become healthy (poll /session) with timeout
$backendUrl = "http://${uiHost}:${bindPort}"
$maxWait = 30  # seconds
$waited = 0
$sessionCheck = $false
while ($waited -lt $maxWait) {
	try {
		$response = Invoke-WebRequest -Uri "$backendUrl/session" -UseBasicParsing -TimeoutSec 3
		if ($response.StatusCode -eq 200) {
			Write-Host "Backend is running at $backendUrl"
			$sessionCheck = $true
			break
		}
	} catch {
		# ignore and retry
	}
	Start-Sleep -Seconds 1
	$waited += 1
}
if (-not $sessionCheck) {
	Write-Host "WARNING: HTTP health-check failed at $backendUrl after $maxWait seconds. Inspecting logs for startup markers..."
	try {
		if (Test-Path $errLog) {
			$tail = Get-Content $errLog -Tail 200 -ErrorAction SilentlyContinue | Out-String
			if ($tail -match "Application startup complete" -or $tail -match "Uvicorn running on") {
				Write-Host "Backend appears to have started (log shows Uvicorn startup). Open the UI at $backendUrl/ui"
				$sessionCheck = $true
			} else {
				Write-Host "ERROR: Backend not responding at $backendUrl after $maxWait seconds. Check logs: $outLog and $errLog"
				Write-Host "Last lines of stderr:"
				Write-Host $tail
			}
		} else {
			Write-Host "ERROR: backend.err.log not found. Check $outLog and process status."
		}
	} catch {
		Write-Host "ERROR while inspecting logs: $_"
	}
} else {
	Write-Host "Backend started and healthy. Open the HTML UI at $backendUrl/ui"
}
