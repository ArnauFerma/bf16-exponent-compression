# Compresion lossless de pesos BF16

Compresion sin perdidas de pesos de modelos de IA, codificando solo el campo
**exponente** de BF16. Misma familia que [DFloat11](https://github.com/LeanModels/DFloat11)
(NeurIPS 2025). Signo y mantisa se guardan crudos porque su entropia ya es
practicamente maxima; todo el margen esta en el exponente (2,63 bits de 8).

## Estado en una linea

La hipotesis original del proyecto (un **codigo por escalones** con tabla de
10 B en vez de la LUT jerarquica de 4 KiB de Huffman) **ha sido refutada**: es
0,85 puntos peor en compresion *y* entre 5% y 43% mas lenta. Pero el trabajo
produjo un codec bastante mejor por otra via.

## Mejor configuracion medida

**Huffman, BLOCK=64, 128 hilos, salida en shared, indice de 8 bits**

| | Fase 2 (punto de partida) | ahora |
|---|---|---|
| tiempo (64M simbolos, GTX 1050 Ti) | 9,70 ms | **4,82 ms** |
| compresion | 30,73% | **32,97%** |

2,01x mas rapido y +2,24 puntos, con dos cambios que **no tocan el codigo de
entropia**: coalescer la escritura de salida, y sustituir el indice de offsets
uint32 por longitudes de 8 bits con prefix-sum de warp.

## Los documentos

| Fichero | Que contiene |
|---|---|
| **[HANDOFF.md](HANDOFF.md)** | **Empezar aqui.** Estado actual, que esta medido y que no, y que hacer a continuacion. |
| [RESULTS.md](RESULTS.md) | Registro cronologico: Fase 1 (CPU), Fase 2 (kernels), 2b (patron de acceso), 2c (vectorial e indice). Todas las tablas de numeros. |
| [ALQUILER_GPU.md](ALQUILER_GPU.md) | Como y donde alquilar una GPU por horas para las medidas que faltan. |
| [INSTRUCCIONES_RTX4070_WIN.md](INSTRUCCIONES_RTX4070_WIN.md) | Guia para un operador con una RTX 4070 en Windows 11. |
| [INSTRUCCIONES_LINUX.md](INSTRUCCIONES_LINUX.md) | Lo mismo para Linux. |
| [INSTRUCCIONES_RTX3060.md](INSTRUCCIONES_RTX3060.md) | Lo mismo para una RTX 3060 en Windows. |

## Reproducir desde cero

```bash
# Linux / GPU alquilada
bash setup_cloud.sh      # o setup_linux.sh en una maquina propia
bash run_all.sh
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

`setup_*` instala dependencias, descarga Qwen3-0.6B, extrae los pesos y
**verifica que los kernels descomprimen bit a bit**. Si esa verificacion falla
el script sale con error a proposito: los tiempos de un descompresor
incorrecto no valen nada.

`run_all` ejecuta las cinco etapas de medida y empaqueta los resultados.

## Codigo

| Fichero | Que es |
|---|---|
| `ladder_codec.py`, `df11_reference.py` | Codecs de referencia en CPU (escalera y Huffman canonico). |
| `bitpack.py` | Encoder vectorizado; produce un bitstream **byte a byte identico** al de referencia. |
| `extract_real_weights.py` | Extrae pesos BF16 crudos de un `.safetensors` sin necesitar torch. |
| `gpu_kernels.py` | Los dos kernels base. |
| `kernel_opt.py` | Variantes con puesta en shared de entrada/salida. |
| `kernel_idx8.py` | Indice de 8 bits + prefix-sum de warp. |
| `bench_*.py` | Bancos de medida. |
| `analysis_*.py` | Entropia conjunta, informacion mutua, coste de cada esquema de indice. |

## Requisitos

- GPU NVIDIA con capacidad de computo >= 6.1 (para perfilar con Nsight
  Compute hace falta >= 7.0: **Pascal no sirve**)
- Python 3.10+
- ~5 GB de disco
