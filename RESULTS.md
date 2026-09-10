# Fase 1 — Validación en CPU: resultados

Ejecutado sobre pesos reales de **Qwen/Qwen3-0.6B** (`model.safetensors`,
751.632.384 pesos BF16 nativos, `torch_dtype: bfloat16` confirmado en
`config.json`). Todos los criterios de la Fase 1 del HANDOFF se cumplen.

## Criterios (Sección 5, Fase 1)

| Paso | Criterio | Resultado |
|---|---|---|
| 2. Entropía del exponente | entre 2,4 y 3,0 bits | **2,634 bits** ✓ |
| 3. Huffman canónico (`df11_reference.py`) | ≥28% ahorro, roundtrip exacto | **32,56%** ahorro, roundtrip bit a bit **SI** ✓ |
| 4. Reasignar escalera a la distribución real | — | hecho, 43 símbolos distintos (vs 25 en sintético) |
| 5. Codec de escalera | roundtrip exacto, <1 punto de Huffman | roundtrip **SI**, gap = **0,85 puntos** ✓ |

## Comparativa Huffman vs escalera (BLOCK=256, índice uint32)

| | Sintético (seed 1234) | Real (Qwen3-0.6B) |
|---|---|---|
| pesos | 2.162.688 | 751.632.384 |
| exponentes distintos | 25 | 43 |
| entropía exponente | 2,7182 bits | 2,6340 bits |
| Huffman canónico | 31,98% | **32,56%** |
| Escalera (1,1,1,2) | 31,28% (−0,69 pts) | **31,72%** (−0,85 pts) |
| Escalera (1,1,2,2) | 30,92% (−1,05 pts) | 31,52% (−1,05 pts) |
| Escalera (1,2,2,3) | 29,95% (−2,03 pts) | 30,18% (−2,38 pts) |

La forma **(1,1,1,2)** del handoff sigue siendo la mejor de las tres
probadas, en sintético y en real. Los pesos reales están *más*
concentrados (entropía más baja, pero más símbolos distintos de cola
larga) que el sintético — el diseño generaliza sin cambios de forma,
solo cambia qué exponente ocupa cada ranura (la cabecera de 10 bytes).

## Barrido de `BLOCK` (datos reales, escalera (1,1,1,2))

| BLOCK | bloques | índice | % overhead | reducción total |
|---|---|---|---|---|
| 64 | 11.744.256 | 44,8 MB | 4,43% | 29,37% |
| 128 | 5.872.128 | 22,4 MB | 2,26% | 30,93% |
| 256 | 2.936.064 | 11,2 MB | 1,14% | 31,72% |
| 512 | 1.468.032 | 5,6 MB | 0,58% | 32,11% |
| 1024 | 734.016 | 2,8 MB | 0,29% | 32,30% |

Confirma el trade-off del handoff: subir `BLOCK` reduce el overhead de
índice pero alarga la cadena serie de decodificación dentro de cada
bloque (más símbolos a decodificar en secuencia por hilo/warp).

## Metodología (importante para no repetir trabajo)

- **Tamaño comprimido**: calculado **analíticamente** (`Σ counts[s] · length[s]`)
  sobre los conteos exactos de exponente del fichero completo — no es una
  estimación, es exacto, pero no pasa por construir el bitstream de 751M
  símbolos en Python puro (inviable en tiempo razonable sin GPU/Cython).
- **Roundtrip bit a bit**: verificado sobre una **muestra aleatoria de
  2.000.000 símbolos** (con reemplazo, `outputs/real_weights_bf16.bin`
  vía `np.memmap` para no cargar 1,5 GB en RAM), usando la tabla de
  códigos derivada de la distribución **completa**. Un código de prefijo
  es memoryless por símbolo, así que esto prueba la corrección del codec
  igual de bien que codificar el fichero entero — la muestra golpeó el
  camino de escape 9.652 veces, así que esa rama también quedó cubierta.
- `real_exp_counts.npy` guarda los conteos exactos de los 751,6M pesos
  para no tener que releer el `.safetensors` (1,4 GB) en próximas
  ejecuciones.

## Descartado / observado

- El entorno tiene poca RAM (3,5 GiB + 3,5 GiB swap). Cargar el
  `.safetensors` completo (`f.read()` de 1,5 GB) o concatenar un array de
  751M `uint16` de una sola vez mata el proceso (OOM, exit 137). Hubo que
  extraer tensor a tensor en streaming, escribiendo directo a disco y
  acumulando conteos incrementalmente.

## Bloqueado — requiere GPU

Este entorno no tiene GPU (`nvidia-smi` no existe, no hay `torch`
instalable con CUDA). **Fase 2 (kernel GPU, Nsight Compute) y Fase 3
(integración con matmul, tokens/s) no se pueden ejecutar aquí.** La
pregunta que decide todo el proyecto — "¿el kernel de exponente está
limitado por memoria o por cómputo?" — sigue sin respuesta. Si el
objetivo es continuar, hace falta una máquina con GPU (o comparar
directamente contra la implementación publicada de DFloat11 en
github.com/LeanModels/DFloat11, que sí trae kernels).

## Ficheros nuevos en este directorio

| Fichero | Qué es |
|---|---|
| `ladder_codec.py` | Codec de escalera genérico (cualquier `rung_bits`), con codificación/decodificación de bitstream real y verificación de Kraft. Incluye auto-test. |
| `bench.py` | Banco de pruebas: Huffman vs escalera, sintético y real, barridos de forma y `BLOCK`, roundtrip por muestreo. Genera `outputs/bench_results.json`. |
| `outputs/real_weights_bf16.bin` | 751.632.384 pesos BF16 crudos extraídos de Qwen3-0.6B (todos los tensores, 1,4 GB). |
| `outputs/real_exp_counts.npy` | Conteos exactos por exponente (256,) del fichero anterior. |
| `outputs/bench_results.json` | Resultados numéricos completos de `bench.py`. |
| `outputs/bench_log.txt` | Log de consola de la última ejecución de `bench.py`. |
| `real_model/` | `model.safetensors` + `config.json` de Qwen3-0.6B descargados de Hugging Face. |
