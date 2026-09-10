# Ejecutar en otra maquina Windows con GPU

Preparado para una **RTX 3060**, pero sirve para cualquier NVIDIA con
capacidad de computo >= 7.0 (Turing/Ampere/Ada). Tiempo estimado: ~15
minutos, casi todo descarga.

---

## Que se esta probando, exactamente

No se busca "un numero mejor". Se busca **falsar una explicacion**.

En la GTX 1050 Ti el rendimiento se hunde al subir `BLOCK` (9,7 ms con
BLOCK=64 -> 154 ms con BLOCK=1024). La explicacion propuesta es que el
working set residente desborda la L2. La metrica que manda no es el tamano
de la L2 sino **la L2 por hilo residente**:

| | L2 | SMs | hilos/SM | hilos residentes | **L2 / hilo** |
|---|---|---|---|---|---|
| GTX 1050 Ti | 1 MB | 6 | 2048 | 12.288 | **85 B** |
| RTX 3060 | 3 MB | 28 | 1536 | 43.008 | **73 B** |
| RTX 4090 | 72 MB | 128 | 1536 | 196.608 | 384 B |

La 3060 tiene 3x mas L2 pero 3,5x mas hilos residentes: por hilo esta
**algo peor** que la 1050 Ti.

> ### Prediccion, escrita antes de medir
> En la RTX 3060 el precipicio de `BLOCK` debe aparecer **en el mismo sitio**
> — entre 128 y 256 — pese a 3x la L2, 4,7x los SMs y 3,2x el ancho de banda.
> Working set con BLOCK=256: 3,73 MB contra 3 MB de L2. Con BLOCK=128: 1,86 MB,
> cabe.
>
> Si el precipicio sale en el mismo sitio, la explicacion queda confirmada en
> silicio distinto. **Si se mueve, la explicacion es falsa** y hay que
> replantear antes de gastar dinero en alquilar nada.

Y ademas: Nsight Compute **si** soporta Ampere, asi que de paso se obtiene lo
que la 1050 Ti no permite medir — ocupacion real, razones de stall y
divergencia de warp (riesgo #4 del HANDOFF).

---

## 1. Requisitos

- **Python 3.12** — `winget install Python.Python.3.12`
- **Driver NVIDIA** reciente (>= 525)
- **~4 GB de disco** (1,4 GB modelo + 1,4 GB extraido + entorno)
- **Nsight Compute** — `winget install Nvidia.Nsight.Compute`

## 2. Permisos de contadores (importante)

Por defecto Windows restringe los contadores de rendimiento de la GPU. Sin
esto `ncu` falla con `ERR_NVGPUCTRPERM`. Dos opciones:

- **Panel de control de NVIDIA** -> menu *Escritorio* -> *Habilitar contadores
  de rendimiento de la GPU* -> **permitir a todos los usuarios**. Reiniciar.
- O simplemente ejecutar `profile_ncu.ps1` desde una PowerShell **como
  administrador**.

## 3. Copiar y ejecutar

Copia esta carpeta **sin** `.venv\`, `outputs\`, `real_model\` ni `.git\`
(el zip `portable_bundle.zip` ya viene filtrado). Luego:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

Instala dependencias, descarga Qwen3-0.6B, extrae los pesos y **verifica que
los dos kernels decodifican bit a bit correctamente**. Si esa verificacion
falla, para: algo va mal y el resto de medidas no valen.

Despues:

```powershell
.\.venv\Scripts\python.exe bench_gpu.py 64000000 > resultados_bench.txt
powershell -ExecutionPolicy Bypass -File profile_ncu.ps1
```

**Para el benchmark, cierra todo lo que use GPU** (navegadores, WhatsApp,
Discord, reproductores). En la 1050 Ti eso bajo el IQR de ~5 ms a ~0 ms. El
reloj no se puede fijar en tarjetas consumer bajo WDDM, asi que la limpieza
del escritorio es lo unico que hay.

## 4. Que traer de vuelta

- `resultados_bench.txt` — el barrido completo
- `ncu_out\*.csv` — 15 ficheros (5 BLOCK x 3 kernels)
- `outputs\gpu_bench.json`

La metrica decisiva en los CSV es **`lts__t_sector_hit_rate.pct`** (acierto de
L2). Si la explicacion es correcta, debe desplomarse entre BLOCK=128 y
BLOCK=256, en paralelo al hundimiento del rendimiento.

## 5. Si algo falla

| Sintoma | Causa probable |
|---|---|
| `ERR_NVGPUCTRPERM` | permisos de contadores, ver punto 2 |
| `Failed to find CUDA headers` | falto el extra `[ctk]`: `pip install "cupy-cuda12x[ctk]"` |
| DLL de NVRTC, "Windows error 32" | pip todavia tenia el fichero abierto; reintentar |
| `ncu.exe no encontrado` | anadir al PATH `C:\Program Files\NVIDIA Corporation\Nsight Compute <version>\` |
| OOM en la extraccion | no deberia: es streaming tensor a tensor |
