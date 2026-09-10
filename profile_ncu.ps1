# Perfilado con Nsight Compute. Requiere ncu en el PATH y permisos de
# contadores (ver PORTABLE.md).
# Uso:  powershell -ExecutionPolicy Bypass -File profile_ncu.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

$ncu = (Get-Command ncu -ErrorAction SilentlyContinue).Source
if (-not $ncu) {
    $cand = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*" -Filter ncu.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cand) { $ncu = $cand.FullName }
}
if (-not $ncu) { Write-Host "ncu.exe no encontrado. Ver PORTABLE.md" -ForegroundColor Red; exit 1 }
Write-Host "ncu: $ncu" -ForegroundColor Cyan

# lts__t_sector_hit_rate es LA metrica decisiva: si el precipicio de BLOCK es
# por L2, el acierto de L2 debe hundirse entre BLOCK=128 y BLOCK=256.
$metrics = @(
    "lts__t_sector_hit_rate.pct",                                   # acierto L2
    "l1tex__t_sector_hit_rate.pct",                                 # acierto L1
    "dram__bytes.sum.per_second",                                   # ancho de banda real
    "sm__warps_active.avg.pct_of_peak_sustained_active",            # ocupacion lograda
    "smsp__thread_inst_executed_per_inst_executed.ratio",           # divergencia
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
Write-Host "`nHecho. Comprime la carpeta ncu_out\ y resultados_bench.txt y llevatelos." -ForegroundColor Green
