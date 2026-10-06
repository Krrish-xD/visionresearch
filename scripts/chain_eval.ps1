
# ============================================================
# chain_eval.ps1
# Watches InternVL3 CLEVR evaluation, then auto-launches Qwen2.5-VL
# ============================================================

$InternVLOut  = "results\raw_predictions\internvl3-8b_clevr.jsonl"
$TargetLines  = 80000
$PollInterval = 60    # seconds between checks
$Workspace    = $PSScriptRoot | Split-Path -Parent

Set-Location $Workspace

function Get-LineCount($path) {
    if (-not (Test-Path $path)) { return 0 }
    (Get-Content $path -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
}

Write-Host "======================================================"
Write-Host "  Chain Eval: InternVL3-8B → Qwen2.5-VL"
Write-Host "  Target: $TargetLines items in $InternVLOut"
Write-Host "======================================================"

# -- Wait for InternVL3 to finish ----------------------------------
while ($true) {
    $count = Get-LineCount $InternVLOut
    $pct   = [math]::Round($count / $TargetLines * 100, 1)
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$stamp] InternVL3 progress: $count / $TargetLines ($pct%)"

    if ($count -ge $TargetLines) {
        Write-Host ""
        Write-Host "✅ InternVL3 COMPLETE ($count lines). Starting Qwen2.5-VL..."
        break
    }
    Start-Sleep -Seconds $PollInterval
}

# -- Launch Qwen2.5-VL ---------------------------------------------
Write-Host ""
Write-Host "======================================================"
Write-Host "  Launching: Qwen2.5-VL 7B | Dataset: CLEVR 80k"
Write-Host "======================================================"

& ".venv\Scripts\python.exe" scripts\run_universal_predictions.py `
    --models qwen2.5-vl-7b `
    --datasets clevr

Write-Host ""
Write-Host "======================================================"
Write-Host "  ✅ Qwen2.5-VL evaluation COMPLETE"
Write-Host "  Output: results\raw_predictions\qwen2.5-vl-7b_clevr.jsonl"
Write-Host "======================================================"
