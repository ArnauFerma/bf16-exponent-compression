# Nsight Compute profiling. Needs ncu on the PATH and counter permissions
# (see OPERATOR_WINDOWS.md).
# Uso:  powershell -ExecutionPolicy Bypass -File profile_ncu.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

$ncu = (Get-Command ncu -ErrorAction SilentlyContinue).Source
if (-not $ncu) {
    $cand = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*" -Filter ncu.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cand) { $ncu = $cand.FullName }
}
if (-not $ncu) { Write-Host "ncu.exe not found. See OPERATOR_WINDOWS.md" -ForegroundColor Red; exit 1 }
Write-Host "ncu: $ncu" -ForegroundColor Cyan

# lts__t_sector_hit_rate is THE decisive metric: if the BLOCK cliff is an L2
# effect, the L2 hit rate must collapse between BLOCK=128 and BLOCK=256.
$metrics = @(
    "lts__t_sector_hit_rate.pct",                                   # L2 hit rate
    "l1tex__t_sector_hit_rate.pct",                                 # L1 hit rate
    "dram__bytes.sum.per_second",                                   # achieved bandwidth
    "sm__warps_active.avg.pct_of_peak_sustained_active",            # achieved occupancy
    "smsp__thread_inst_executed_per_inst_executed.ratio",           # divergence
    "gpu__time_duration.sum"
) -join ","

New-Item -ItemType Directory -Force ncu_out | Out-Null
foreach ($block in 64,128,256,512,1024) {
    foreach ($codec in "huffman","ladder","floor") {
        $out = "ncu_out\b${block}_${codec}.csv"
        Write-Host "`n-- BLOCK=$block codec=$codec" -ForegroundColor Yellow
        & $ncu --csv --target-processes all `
            --kernel-name "regex:decode_|mem_floor" `
            --launch-skip 1 --launch-count 1 `
            --metrics $metrics `
            $py profile_run.py --block $block --threads 128 --codec $codec `
            > $out 2>&1
        if (Test-Path $out) { Write-Host "  -> $out" }
    }
}
Write-Host "`nDone. run_all.ps1 copies ncu_out\ into results\<gpu>\." -ForegroundColor Green
