# Todas las medidas + perfilado, en Windows.
# Uso:  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$OUT = "resultados"
New-Item -ItemType Directory -Force $OUT | Out-Null

Write-Host "== GPU ==" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.used --format=csv |
    Tee-Object -FilePath "$OUT\gpu_info.txt"

# En Windows/WDDM no se pueden fijar los relojes. Lo unico que ayuda es
# cerrar todo lo que use la GPU antes de empezar.
Write-Host "`nCIERRA navegadores, Discord, Steam, juegos y reproductores antes de seguir." -ForegroundColor Yellow
Write-Host "Pulsa Enter cuando este listo (o espera 20 s)..." -ForegroundColor Yellow
# En una consola no interactiva KeyAvailable lanza excepcion; se ignora.
try {
    $t = 0
    while (-not [Console]::KeyAvailable -and $t -lt 20) { Start-Sleep -Seconds 1; $t++ }
    if ([Console]::KeyAvailable) { [Console]::ReadKey($true) | Out-Null }
} catch { Start-Sleep -Seconds 5 }

Write-Host "`n== 1/5  barrido base (Huffman vs escalera, kernel original) ==" -ForegroundColor Cyan
& $py bench_gpu.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\1_bench_base.txt" | Select-Object -Last 20

Write-Host "`n== 2/5  variantes del patron de acceso (atribucion) ==" -ForegroundColor Cyan
& $py bench_opt.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\2_bench_opt.txt" | Select-Object -Last 25

Write-Host "`n== 3/5  Huffman vs escalera CON optimizacion ==" -ForegroundColor Cyan
& $py bench_head2head.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\3_head2head.txt" | Select-Object -Last 25

Write-Host "`n== 4/5  indice de 8 bits + prefix-sum ==" -ForegroundColor Cyan
& $py bench_idx8.py 64000000 2>&1 | Tee-Object -FilePath "$OUT\4_bench_idx8.txt" | Select-Object -Last 20

Write-Host "`n== 5/5  perfilado con Nsight Compute (opcional) ==" -ForegroundColor Cyan
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
    Write-Host "ncu no encontrado; se salta. Los tiempos de arriba ya responden lo principal." -ForegroundColor Yellow
}

Copy-Item -Force outputs\gpu_bench.json "$OUT\" -ErrorAction SilentlyContinue

Write-Host "`n== empaquetando ==" -ForegroundColor Cyan
if (Test-Path resultados.zip) { Remove-Item resultados.zip -Force }
Compress-Archive -Path $OUT -DestinationPath resultados.zip
Write-Host "LISTO -> resultados.zip   (mandar este fichero)" -ForegroundColor Green
