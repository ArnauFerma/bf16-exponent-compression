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

---

# Fase 2b — Optimizacion del patron de acceso

La Fase 2 dejo el kernel a **8,2% del pico** de ancho de banda. Eso no es
"limitado por memoria" en el sentido util: el kernel suelo tiene el mismo
patron de acceso malo, asi que medía el techo *de ese layout*, no el del
hardware. Aqui se ataca el layout.

## Dos problemas separables

- **Entrada**: cada hilo recorre su propia region de bits -> un warp se
  dispersa en 32 flujos. Pero las regiones de todos los hilos de un bloque
  CUDA son contiguas: se pueden cargar coalescidas a shared.
- **Salida**: 71% del trafico y va peor. Cada hilo escribe `BLOCK` bytes
  consecutivos, asi que un warp escribe 32 bytes separados `BLOCK` entre si:
  32 sectores distintos para 32 bytes de datos. Misma solucion.

Se compilaron **4 variantes** (entrada x salida) para poder atribuir la
mejora en vez de adivinarla. Todas verificadas bit a bit.

## Resultado (BLOCK=64, 128 hilos, 64M simbolos)

| variante | shared | ms | GB/s | vs base |
|---|---|---|---|---|
| base (global) | — | 9,74 | 9,2 | 1,00 |
| smem entrada | 3,7 KB | 9,66 | 9,3 | 1,01x |
| **smem salida** | 8,2 KB | **5,07** | **17,7** | **1,92x** |
| ambas | 11,9 KB | 6,59 | 13,6 | 1,48x |

**La salida sola casi duplica el rendimiento**: 8,2% -> 15,8% del pico.
Coalescer la entrada vale ~1%: nunca fue el cuello de botella.

**Hacer las dos es PEOR que solo la salida** (6,59 vs 5,07). La puesta en
shared de la entrada gasta memoria compartida y anade una barrera sin
comprar nada, y eso se paga en ocupacion. Combinar optimizaciones empeoro.

## En BLOCK=256 se invierte — y eso confirma lo de la L2

| BLOCK=256, 64 hilos | ms | vs base |
|---|---|---|
| base | 57,14 | 1,00 |
| **smem entrada** | **13,05** | **4,38x** |
| smem salida | 15,60 | 3,66x |

Con BLOCK=64/128 el working set de entrada cabe en la L2 de 1 MiB, asi que
ponerlo en shared es redundante (~1%). Con BLOCK=256 son 1040 KiB y ya **no
cabe**: ponerlo en shared vale 4,38x. El cruce cae exactamente donde caia el
precipicio de rendimiento de la Fase 2.

Son ya **dos lineas de evidencia independientes** para la misma explicacion.

## Efecto sobre el compromiso con la compresion

| BLOCK | mejor ms | compresion |
|---|---|---|
| 64 | 5,07 | 29,37% |
| 128 | 9,26 | 30,93% |
| 256 | 13,05 | 31,72% |

Sigue habiendo tension, pero mucho menos brutal: BLOCK=256 era 6x mas lento
que BLOCK=64, ahora es 2,6x a cambio de 2,35 puntos mas de compresion.

## Huffman vs escalera, rehecho con el kernel optimizado

La conclusion de la Fase 2 se midio con el kernel lento. Al mover el cuello
de botella hay que **rehacer** la comparacion, no extrapolarla.

| BLOCK | hilos | variante | huffman ms | escalera ms | l/h |
|---|---|---|---|---|---|
| 64 | 64 | salida | 5,23 | **5,07** | 0,970 |
| **64** | **128** | **salida** | **4,80** | 5,07 | **1,055** |
| 128 | 128 | salida | 8,75 | 9,73 | 1,112 |
| 256 | 128 | salida | 12,73 | 18,20 | 1,430 |
| 256 | 64 | entrada | 13,04 | 13,05 | 1,001 |

**Optimo global: Huffman, BLOCK=64, 128 hilos, salida en shared: 4,80 ms.**
2,02x mas rapido que el mejor de la Fase 2 (9,70 ms).

La optimizacion **refuerza** la conclusion de la Fase 2 en vez de tumbarla.
La escalera ahora pierde en **los dos ejes**: 0,85 puntos peor de compresion
*y* entre 5% y 43% mas lenta. Antes al menos ganaba en el regimen limitado
por computo.

Y este es el regimen donde debia ganar: al 16,5% del pico, el computo pesa
mas que antes, y aun asi pierde. La duda que el propio HANDOFF planteaba
resulta ser la correcta:

> *"Huffman canonico usa LUT: **un** acceso a SRAM. La escalera usa `clz` +
> rama + shift + mask. Contando instrucciones puede perder."*

Pierde. Un acceso a shared bate una cadena dependiente de clz/rama/shift/mask.

## Bug encontrado y corregido

Al ampliar el barrido a BLOCK=512/1024 aparecio salida **incorrecta y no
determinista** en el kernel base de la escalera. Causa: el `return` temprano
estaba **antes** de `__syncthreads()`. Si `n_blocks` no es multiplo de
`blockDim.x`, parte de los hilos del bloque salen y el resto espera en una
barrera que ya no completan todos: comportamiento indefinido.
`decode_huffman` sincronizaba antes de salir, por eso solo fallaba la
escalera.

Corregido; las 20 combinaciones BLOCK x hilos verifican bit a bit. El
barrido de la Fase 2 se **repitio** con el kernel corregido y los numeros no
se mueven (9,71/9,74 frente a 9,70/9,73 en BLOCK=64), asi que las
conclusiones publicadas se mantienen.

## Sin explicar

Con BLOCK=128 y puesta en shared de la entrada, Huffman es **2x** mas rapido
que la escalera (12,83 vs 26,14 ms), mucho mas de lo que justifica el 4,8%
de diferencia de tamano del stream. Ni los registros (21 vs 15) ni la shared
lo predicen. Queda anotado como anomalia, no explicado: no esta en el camino
optimo, pero es el tipo de cosa que a veces esconde un bug.

## Pendiente en hardware con L2 grande

Todo esto es Pascal con 1 MiB de L2. En una RTX 4070 (36 MB de L2, 534 B por
hilo residente, 6,3x mejor) el working set con BLOCK=1024 son 24,5 MB y
**cabe**. Prediccion: el precipicio deberia desaparecer, la puesta en shared
de la entrada deberia dejar de importar, y BLOCK=512-1024 pasaria a ser
viable — la primera configuracion donde coinciden la mejor compresion
(32,3%) y buena velocidad.

---

# Fase 2c — Codificacion vectorial (no) e indice de 8 bits (si)

## Pregunta: sirve la codificacion vectorial / por pares?

La intuicion es razonable: Huffman compra compresion pagando en tamano de
diccionario, y la escalera existe precisamente para no pagar eso. Escalar a
pares deberia ser donde mas se note. Se midio en vez de discutirlo.

La ganancia posible tiene **dos fuentes** que conviene separar:

**(a) Redundancia del codigo.** Huffman esta a 2,5832 bits contra una
entropia de 2,5520: **0,0312 bits/simbolo** de margen. Ningun codificador de
entropia — vectorial, aritmetico, ANS — puede superar esa cota si los
simbolos son independientes.

**(b) Correlacion entre exponentes vecinos.** Esto **no** esta acotado por
(a): seria margen nuevo. Y es una pregunta empirica.

Medido en seis ventanas repartidas por todo el modelo (no solo los primeros
64M pesos, que son `embed_tokens` y podrian enganar):

| offset | H(X) | H(Y dado X) | I(X;Y) |
|---|---|---|---|
| 0 | 2,5494 | 2,5491 | 0,00024 |
| 93.954.048 | 2,5591 | 2,5589 | 0,00026 |
| 187.908.096 | 2,5545 | 2,5543 | 0,00020 |
| 375.816.192 | 2,6535 | 2,6435 | 0,00998 |
| 563.724.288 | 2,6493 | 2,6406 | 0,00870 |
| 711.632.384 | 2,6436 | 2,6347 | 0,00884 |

**Informacion mutua maxima: 0,00998 bits.** Los exponentes vecinos son
independientes a efectos practicos. El modelado por contexto de orden 1 lo
confirma: **+0,0000 bits/simbolo**.

Asi que solo queda (a), y Huffman sobre pares captura la mitad: 2,5668 vs
2,5832, o sea 0,0164 bits/simbolo — **0,1% del fichero total**.

**Y para la escalera es peor que inutil:**

| escalera sobre pares | bits/simbolo | tabla |
|---|---|---|
| (1,1,1,2) | 5,1172 | 20 B |
| (2,2,3,4) | 3,1712 | 64 B |
| **(4,4,5,6)** | **2,7542** | 256 B |
| (5,5,6,7) | 3,0699 | 512 B |
| (6,6,7,8) | 3,5106 | 842 B |

La mejor escalera sobre pares (2,7542) es **peor que la escalera escalar**
(2,7089) con una tabla 25x mayor. La geometria de escalones funciona porque
tres simbolos concentran el 69% de la masa; al repartir sobre 421 pares
observados la distribucion se aplana y los escalones en potencias de dos no
la siguen. La codificacion vectorial es justo la tecnica que infla tablas,
que es lo unico que la escalera existe para evitar.

**Donde la intuicion si acierta:** con modelado por contexto, Huffman
necesitaria 31 x 4 KiB = **124 KiB de tablas, que no caben en los 48 KiB de
shared**; la escalera necesitaria **310 B**. Si hubiera correlacion, la
escalera seria la unica opcion viable en GPU. El argumento estructural es
correcto; lo que no existe es la redundancia que explotar.

**Un angulo si sobrevive, pero para Huffman:** codificar pares reduce a la
mitad el numero de iteraciones de decodificacion, y por tanto la cadena
serie de cargas dependientes, que es el cuello de botella real. Huffman
sobre pares es a la vez 0,0164 bits/simbolo mas pequeno y la mitad de
iteraciones, a cambio de una LUT de 8 KiB. Pendiente de probar.

## Indice de 8 bits + prefix-sum de warp

La longitud de bloque tiene poco rango: con BLOCK=64, entre 130 y 273 bits
(rango 143 < 256). Cabe en 8 bits. Entonces, en vez de un offset absoluto
uint32 por bloque, se guarda un uint32 por superbloque de 32 bloques (un
warp) mas **un uint8 por bloque con su longitud**, y el offset se recupera
con un prefix-sum exclusivo dentro del warp (5 pasos de `__shfl_up_sync`).

| BLOCK | codec | indice | ms | B/bloque | compresion | |
|---|---|---|---|---|---|---|
| 64 | huffman | uint32 | 4,81 | 4,000 | 30,73% | |
| **64** | **huffman** | **uint8+ps** | **4,82** | **1,125** | **32,97%** | **+2,25 pts** |
| 64 | escalera | uint32 | 5,07 | 4,000 | 29,94% | |
| 64 | escalera | uint8+ps | 5,11 | 1,125 | 32,19% | +2,25 pts |
| 128 | huffman | uint32 | 8,74 | 4,000 | 32,29% | |
| 128 | huffman | uint8+ps | 8,77 | 1,125 | 33,41% | +1,12 pts |

**El prefix-sum no cuesta nada medible** (4,82 vs 4,81 ms, dentro del ruido):
cinco `__shfl_up_sync` no se notan al lado de una cadena de 64 cargas
dependientes. El trafico de indice baja de 4 MB a 1,125 MB.

### Desaparece la tension velocidad/compresion

Era el problema estructural de toda esta fase. Proyectado al modelo completo:

| | config mas rapida | mejor compresion |
|---|---|---|
| antes | BLOCK=64 -> 30,22% | BLOCK=1024 -> 33,15%, pero **32x mas lento** |
| despues | BLOCK=64 -> **32,46%** | BLOCK=1024 -> 33,15% |

La distancia entre "rapido" y "comprime bien" pasa de 2,93 puntos a 0,69.
Ya no hay que elegir.

**Mejor configuracion global: Huffman, BLOCK=64, 128 hilos, salida en shared,
indice de 8 bits — 4,82 ms y 32,97%.** Frente a la referencia de la Fase 2
(9,70 ms, 30,73%): **2,01x mas rapido y +2,24 puntos**, con dos cambios que
no tocan ni el codigo de entropia ni el bucle de decodificacion.

### Limitacion

El indice de 8 bits **solo llega hasta BLOCK=128**. Con BLOCK=256 el rango de
longitud de bloque es 409, por encima de los 255 que caben en un delta de 8
bits; el constructor lanza excepcion en vez de truncar en silencio. Para
bloques mayores hace falta la variante de 16 bits relativos (2,016 B/bloque,
+1,94 puntos con BLOCK=256).

## Lectura de conjunto

Los dos cambios **estructurales** valen 2,24 puntos y 2x de velocidad. La
pregunta sobre el codigo de entropia alrededor de la cual se construyo el
proyecto vale 0,85 puntos, y en contra. El indice y el layout de memoria
importaban muchisimo mas que Huffman-contra-escalera.

La codificacion de entropia esta, a efectos practicos, terminada: Huffman se
queda a 0,031 bits del suelo teorico y no hay correlacion que explotar. Todo
el margen que queda es estructural.

## Pendiente

- **Salida BF16 fusionada**: hoy el kernel emite un byte de exponente por peso
  y hace falta una pasada aparte para mezclar signo+mantisa (64 + 64 MB
  leidos, 128 MB escritos = 256 MB, casi 3x el propio kernel). Fusionarla
  baja el trafico total del pipeline de ~345 MB a ~217 MB (-37%) y quita un
  lanzamiento entero. Necesario para la Fase 3 de todos modos.
- **Huffman sobre pares**: mitad de iteraciones en la cadena serie.

## Ficheros nuevos

| Fichero | Que es |
|---|---|
| `kernel_opt.py` | Kernels con puesta en shared de entrada/salida (4 variantes, para atribuir). |
| `kernel_idx8.py` | Indice de 8 bits + prefix-sum de warp, para Huffman y escalera. |
| `bench_opt.py` | Atribucion de la mejora del patron de acceso. |
| `bench_head2head.py` | Huffman vs escalera con el kernel optimizado. |
| `bench_idx8.py` | uint32 vs uint8+prefix-sum: tiempo y compresion. |
| `analysis_vector.py` | Entropia conjunta, informacion mutua, Huffman/escalera sobre pares. |
| `analysis_index.py` | Informacion mutua por ventanas y coste de cada esquema de indice. |
