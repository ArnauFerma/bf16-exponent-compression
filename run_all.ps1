# All measurements + profiling, on Windows.
# Uso:  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
# results\<gpu-name>\, e.g. results\gtx1050ti\
$name = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
$slug = (($name -replace 'NVIDIA |GeForce |Tesla ', '') -replace '[^A-Za-z0-9]', '').ToLower()
if (-not $slug) { $slug = "unknown-gpu" }
$OUT = "results\$slug"
New-Item -ItemType Directory -Force $OUT | Out-Null

Write-Host "== environment ==" -ForegroundColor Cyan
& $py env_info.py "$OUT\env_info.json"

Write-Host "`n== GPU ==" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.used --format=csv |
    Tee-Object -FilePath "$OUT\gpu_info.txt"

# Under Windows/WDDM the clocks cannot be locked. The only thing that helps
# is closing everything that uses the GPU before starting.
Write-Host "`nCLOSE browsers, Discord, Steam, games and media players before continuing." -ForegroundColor Yellow
Write-Host "Press Enter when ready (or wait 20 s)..." -ForegroundColor Yellow
# In a non-interactive console KeyAvailable throws; ignored.
try {
    $t = 0
    while (-not [Console]::KeyAvailable -and $t -lt 20) { Start-Sleep -Seconds 1; $t++ }
    if ([Console]::KeyAvailable) { [Console]::ReadKey($true) | Out-Null }
} catch { Start-Sleep -Seconds 5 }

Write-Host "`n== 1/5  base sweep (Huffman vs ladder, original kernel) ==" -ForegroundColor Cyan
& $py bench_gpu.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\1_bench_base.txt" | Select-Object -Last 20

Write-Host "`n== 2/5  access-pattern variants (attribution) ==" -ForegroundColor Cyan
& $py bench_opt.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\2_bench_opt.txt" | Select-Object -Last 25

Write-Host "`n== 3/5  Huffman vs ladder WITH the optimisation ==" -ForegroundColor Cyan
& $py bench_head2head.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\3_head2head.txt" | Select-Object -Last 25

Write-Host "`n== 4/5  8-bit index + prefix-sum ==" -ForegroundColor Cyan
& $py bench_idx8.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\4_bench_idx8.txt" | Select-Object -Last 20

Write-Host "`n== 5/5  Nsight Compute profiling (optional) ==" -ForegroundColor Cyan
$ncu = (Get-Command ncu -ErrorAction SilentlyContinue).Source
if (-not $ncu) {
    $c = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*" -Filter ncu.exe -Recurse -ErrorAction SilentlyContinue |
         Select-Object -First 1
    if ($c) { $ncu = $c.FullName }
}
if ($ncu) {
    powershell -ExecutionPolicy Bypass -File profile_ncu.ps1
    if (Test-Path ncu_out) { Copy-Item -Recurse -Force ncu_out "$OUT\" }
} else {
    Write-Host "ncu not found; skipping. The timings above already answer the main question." -ForegroundColor Yellow
}

Copy-Item -Force outputs\gpu_bench.json "$OUT\" -ErrorAction SilentlyContinue

Write-Host "`n== packaging ==" -ForegroundColor Cyan
if (Test-Path results.zip) { Remove-Item results.zip -Force }
Compress-Archive -Path "$OUT\*" -DestinationPath results.zip
Write-Host "DONE -> results.zip   (send this file)" -ForegroundColor Green
