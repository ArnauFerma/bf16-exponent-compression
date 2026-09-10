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

> **Resuelto en la Fase 2** (ver mas abajo): ejecutada en una GTX 1050 Ti.
> Lo de esta seccion describe el entorno de la Fase 1, no el actual.

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

---

# Fase 2 — Kernels GPU: resultados

Ejecutado en una **NVIDIA GeForce GTX 1050 Ti** (Pascal, SM 6.1, 6 SMs,
4 GB, L2 de 1 MiB, pico teorico 112,1 GB/s), sobre los mismos pesos reales
de Qwen3-0.6B (muestra de 64.000.000 de exponentes).

## Que se implemento

| Fichero | Que es |
|---|---|
| `gpu_kernels.py` | Los dos kernels: `decode_huffman` (LUT jerarquica de 4 KiB en shared) y `decode_ladder` (clz + switch, tabla de 10 B). |
| `bitpack.py` | Encoder vectorizado en numpy. Produce un bitstream **byte a byte identico** al del `BitWriter` original (verificado contra ambos codecs, offsets incluidos); el bucle Python era inviable a escala de kernel. |
| `verify_kernels.py` | Correccion: ambos kernels decodifican 32M de exponentes reales **bit a bit** correctos. |
| `bench_gpu.py` | Medida, incluido el kernel `mem_floor`. |

Ambos kernels comparten indice grueso, mapeo un-hilo-por-bloque, el lector
de ventana `peek32` y el layout de salida. **Lo unico que cambia es el
decodificador de simbolo.**

Consumo de recursos (de `kernel.attributes`):

| | registros/hilo | shared |
|---|---|---|
| `decode_huffman` | 21 | 4096 B |
| `decode_ladder` | 15 | 16 B |

## Metodologia

- **Kernel suelo de memoria (`mem_floor`)**: mueve exactamente el mismo
  trafico (lee las mismas palabras del bitstream, escribe los mismos bytes)
  pero **no decodifica**. Es un techo de rendimiento empirico: acercarse a
  el significa que manda la memoria y el codigo de entropia es irrelevante.
- Esta GPU **no permite fijar relojes** (consumer bajo WDDM, y ademas mueve
  el escritorio: oscila entre 139 y 1923 MHz). Se compensa con calentamiento
  sostenido por configuracion, muestreo del reloj, orden de configuraciones
  aleatorizado y mediana de 11 repeticiones. Con el escritorio despejado el
  IQR queda en 0-3 ms.
- El indice de bloques es **uint32**, como dice la spec del formato.

## Resultados (64M simbolos, mediana de 11)

| BLOCK | hilos | huffman ms | escalera ms | suelo ms | h/suelo | l/suelo | **l/h** |
|---|---|---|---|---|---|---|---|
| **64** | 128 | **9,70** | **9,73** | 9,57 | 1,01 | 1,02 | **1,003** |
| 128 | 128 | 51,08 | 34,55 | 12,77 | 4,00 | 2,71 | 0,676 |
| 256 | 128 | 82,75 | 56,95 | 13,31 | 6,22 | 4,28 | 0,688 |
| 512 | 128 | 118,63 | 89,84 | 16,06 | 7,38 | 5,59 | 0,757 |
| 1024 | 128 | 153,77 | 119,27 | 19,52 | 7,88 | 6,11 | 0,776 |

## La respuesta a la pregunta que decide todo

**En el unico punto de operacion que elegirias, el kernel esta limitado por
MEMORIA, y la escalera no aporta nada.**

Hay dos regimenes y apuntan en direcciones opuestas:

- **BLOCK=64 es la configuracion mas rapida, por 3-16x.** Ahi los dos
  decodificadores estan a **1-2% del suelo de memoria** y a **0,3-0,9% el
  uno del otro**. El codigo de entropia es irrelevante. Peor: la escalera es
  sistematicamente la **mas lenta** (1,003-1,009x), que es justo lo que cabe
  esperar — en regimen limitado por memoria su bitstream un 4,8% mas grande
  cuesta tiempo y no compra nada.
- **BLOCK>=128** la escalera gana un 5-32%, confirmando la hipotesis de
  limitado-por-computo... pero **todas esas configuraciones son 3-13x mas
  lentas en absoluto**. Nadie las usaria.

Es decir: **se materializa el riesgo #1 del handoff.** En este hardware la
escalera queda como una curiosidad 0,85 puntos peor en compresion y sin
compensacion en velocidad. El handoff decia explicitamente que ese seria un
resultado valido; lo es.

## Tres matices que acotan la conclusion

**1. "Limitado por memoria" aqui significa limitado por un patron de acceso
malo, no por el hardware.** El pico son 112,1 GB/s; la mejor configuracion
logra **9,1 GB/s, un 8,2% del pico**. Cada hilo recorre su propia region, asi
que un warp se dispersa en 32 flujos independientes. Hay ~10x de margen
encima de la mesa, disponible para **los dos** codecs, atacando el patron de
acceso (decodificacion cooperativa a nivel de warp, cargas vectorizadas).
Ese premio es mucho mayor que los 0,85 puntos que separan a los dos codigos.

**2. La salida domina el trafico.** Con BLOCK=64 se mueven 88,7 MB, de los
cuales **64 MB (71%) son la salida** de exponentes. Los dos bitstreams se
diferencian en menos de 1 MB. Estructuralmente el codigo de entropia solo
puede influir en ~23% del trafico, y fusionar la mezcla de signo+mantisa
para emitir BF16 directamente reduciria aun mas esa fraccion. Por eso los
dos codecs convergen, y esto **no** es especifico de esta GPU.

**3. El precipicio de BLOCK parece un efecto de L2, y por tanto no
transfiere.** La L2 son 1024 KiB, y el working set residente la cruza entre
BLOCK=128 (520 KiB) y BLOCK=256 (1040 KiB) — justo donde se hunde el
rendimiento. Es **consistente, no probado**: BLOCK=128 ya se degrada aunque
todavia quepa. Una H100 tiene ~50 MB de L2 y una RTX 4090 ~72 MB, asi que el
precipicio se desplaza mucho a la derecha, BLOCK grande pasa a ser viable, y
**ese es precisamente el regimen donde gana la escalera**. Este resultado
argumenta en contra de la escalera en Pascal; **no zanja la pregunta en una
tarjeta moderna**.

## No medible aqui

**Nsight Compute no soporta esta GPU.** NVIDIA retiro el soporte de Pascal
(SM 6.x) en la version 2020.1; ninguna version actual perfila una 1050 Ti.
El unico perfilador con soporte Pascal es `nvprof` legacy, que solo viene con
CUDA Toolkit <=11.x (~3 GB de instalacion) y necesita permisos de admin para
los contadores. Se decidio no instalarlo: el kernel suelo responde a la
pregunta central de forma mas directa que los contadores, y el riesgo #4 del
handoff (divergencia de warp en la rama de escape) es discutible cuando en el
punto de operacion no manda el computo.

Queda por tanto **sin medir**: ocupacion real, razones de stall de warp y
divergencia.

## Proximos pasos sugeridos

1. **Local, gratis:** atacar el patron de acceso (8,2% del pico). Si se cierra
   aunque sea la mitad de ese 10x, todas las conclusiones posteriores cambian.
2. **~5 USD, media jornada:** alquilar una RTX 4090 o L40S. Rehacer el barrido
   de BLOCK con L2 grande y usar Nsight Compute para stalls y divergencia.
3. **Solo si 2 es prometedor:** A100/H100 para comparar cara a cara contra los
   kernels publicados de DFloat11 en su hardware objetivo.
