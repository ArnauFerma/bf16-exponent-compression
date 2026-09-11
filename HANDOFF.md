# Compresion lossless de pesos BF16 — handoff

Documento de estado. Ultima actualizacion: **2026-09-11**.

Para el registro cronologico con todas las tablas, ver [RESULTS.md](RESULTS.md).
La version original de este handoff (antes de tener GPU) esta en el historial
de git, commit `098cf3b`.

---

## 0. Resumen en una frase

La hipotesis original — que un **codigo por escalones** con tabla de 10 B
rendiria mejor que la LUT jerarquica de 4 KiB de Huffman — **esta refutada**:
pierde en los dos ejes. Pero atacando el patron de acceso a memoria y el
indice de bloques, el codec pasa de 9,70 ms / 30,73% a **4,82 ms / 32,97%**.

---

## 1. Estado de las afirmaciones

Esto importa mas que nada: separa lo medido de lo especulado.

### Medido y reproducible

Salvo indicacion, sobre **Qwen3-0.6B real** (751.632.384 pesos BF16) y, para
los kernels, una muestra de 64M exponentes en una **GTX 1050 Ti**.

| Afirmacion | Valor | Como se verifico |
|---|---|---|
| Entropia del exponente | 2,634 bits | Conteo exacto sobre 751,6M pesos |
| Huffman canonico | 2,665 bits/exp -> 32,56% | Codec implementado, roundtrip bit a bit |
| Escalera (1,1,1,2) | 2,800 bits/exp -> 31,72% | Idem, −0,85 puntos |
| Ambos kernels GPU | roundtrip bit a bit correcto | 32M simbolos reales, 20 combinaciones BLOCK x hilos |
| **El decodificador NO esta limitado por computo** | 8,2% del pico de ancho de banda | Kernel "suelo de memoria" con el mismo trafico |
| Coalescer la **salida** | **1,92x** | 4 variantes compiladas para atribuir |
| Coalescer la **entrada** | 1,01x | Nunca fue el cuello de botella |
| Precipicio de BLOCK = efecto de L2 | cruce entre 128 y 256 | Coincide con el desbordamiento del working set, y con que la entrada en shared pase a valer 4,38x |
| **Escalera mas lenta que Huffman** | 5% a 43% | Con el kernel optimizado, en todas las configuraciones |
| Informacion mutua entre exponentes vecinos | **0,0002–0,0100 bits** | 6 ventanas repartidas por el modelo |
| Modelado por contexto orden 1 | **+0,0000 bits/simbolo** | No hay correlacion que explotar |
| Escalera sobre pares | 2,754 bits, peor que escalar (2,709) | Con tabla 25x mayor |
| **Indice de 8 bits + prefix-sum** | **+2,25 puntos, coste nulo** | 4,82 vs 4,81 ms |

### No medido — sigue abierto

| Pregunta | Por que importa |
|---|---|
| Que pasa en una GPU con L2 grande | Todo lo anterior es Pascal con 1 MiB de L2. Ver seccion 4. |
| Ocupacion real, stalls de warp, divergencia | Nsight Compute **no soporta Pascal**; no se ha podido medir aqui |
| Comparacion contra los kernels publicados de DFloat11 | Solo se ha comparado contra implementacion propia |
| tokens/s extremo a extremo (Fase 3) | "Kernel mas rapido" no es lo mismo que "inferencia mas rapida" |
| Anomalia: con BLOCK=128 y entrada en shared, Huffman es 2x mas rapido que la escalera | Mucho mas de lo que justifica el 4,8% de diferencia de stream. Sin explicar. |

### Descartado (callejones sin salida ya recorridos)

- **Comprimir el flujo de bits crudo en grupos de 3 o 4 bits.** Entre −18% y
  +12,6%. Muy por debajo de atacar el exponente.
- **Indice de offsets por peso.** Cuesta ~800% del fichero.
- **Prefix-sum global de offsets.** Problema huevo-gallina; la solucion es el
  indice grueso por bloques.
- **Codigos no libres de prefijo.** Cualquier tabla nueva debe pasar Kraft <= 1.
- **Codificacion vectorial / por pares.** Medido: no hay correlacion que
  explotar (I < 0,01 bits) y para la escalera es peor que el codigo escalar.
- **La escalera como via de rendimiento.** Pierde en compresion y en velocidad.

---

## 2. El formato, tal como esta ahora

BF16 = `[1 signo][8 exponente][7 mantisa]`. Signo y mantisa se guardan crudos,
empaquetados en 1 byte por peso. Solo se codifica el exponente.

```
cabecera        : magic, version, n_pesos, BLOCK, tabla de codigos
sign_mantisa    : n_pesos bytes, crudos
exponentes      : bitstream Huffman canonico
indice          : ver abajo
```

**Indice (cambiado en la Fase 2c).** En vez de un offset absoluto uint32 por
bloque (4 B/bloque):

- un **uint32 por superbloque** de 32 bloques (= un warp) -> 0,125 B/bloque
- un **uint8 por bloque** con su longitud en bits, menos un minimo global
  -> 1,0 B/bloque

Total **1,125 B/bloque**. El offset de cada bloque se recupera con un
prefix-sum exclusivo dentro del warp (5 pasos de `__shfl_up_sync`, coste no
medible). Cabe en 8 bits porque la longitud de bloque tiene poco rango: con
BLOCK=64, de 130 a 273 bits.

**Limitacion:** solo llega hasta BLOCK=128. Con BLOCK=256 el rango es 409 y no
cabe; haria falta la variante de 16 bits relativos (2,016 B/bloque).

---

## 3. Lo que aprendio el proyecto

Vale la pena decirlo explicitamente, porque es lo contrario de lo que se
esperaba:

> Los dos cambios **estructurales** (patron de acceso e indice) valen **2x de
> velocidad y +2,24 puntos**. La pregunta sobre el **codigo de entropia**,
> alrededor de la cual se construyo el proyecto entero, vale 0,85 puntos, y en
> contra.

La codificacion de entropia esta, a efectos practicos, terminada: Huffman se
queda a 0,031 bits del suelo teorico y no hay correlacion que explotar. Todo
el margen que queda es estructural.

La duda que el handoff original ya planteaba resulto ser la correcta:

> *"Huffman canonico usa LUT: un acceso a SRAM. La escalera usa `clz` + rama +
> shift + mask. Contando instrucciones puede perder."*

Pierde.

---

## 4. Lo siguiente, por orden de valor

### 4.1 Medir en una GPU con L2 grande — BLOQUEANTE para publicar nada

Todo lo medido es Pascal con 1 MiB de L2. La cifra que gobierna el precipicio
de BLOCK es la **L2 por hilo residente**:

| tarjeta | L2 | SMs | hilos residentes | **L2 / hilo** |
|---|---|---|---|---|
| GTX 1050 Ti (la de casa) | 1 MB | 6 | 12.288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43.008 | 73 B |
| A100 80GB | 40 MB | 108 | 221.184 | 190 B |
| H100 SXM | 50 MB | 132 | 270.336 | 194 B |
| RTX 4090 | 72 MB | 128 | 196.608 | 384 B |
| RTX 4070 | 36 MB | 46 | 70.656 | 534 B |

> **Prediccion registrada antes de medir:** en una tarjeta con L2 grande el
> precipicio de BLOCK **deberia desaparecer**; con BLOCK=1024 el working set
> residente son 24,5 MB (cabe en 36 MB de una 4070, no cabe en 1 MB). La
> entrada en shared deberia dejar de importar, y BLOCK=512-1024 deberia pasar
> a ser viable: la primera configuracion donde coinciden la mejor compresion y
> buena velocidad.
>
> **Si el precipicio sigue ahi, la explicacion de la L2 es falsa** y hay que
> reescribir la Fase 2b antes de que nada de esto salga del repo.

Como hacerlo: [ALQUILER_GPU.md](ALQUILER_GPU.md) (~1,40 USD, menos de una
hora) o las guias para operador con maquina prestada.

### 4.2 Salida BF16 fusionada

Hoy el kernel emite un byte de exponente por peso y hace falta una pasada
aparte para mezclar signo+mantisa: 64 MB + 64 MB leidos y 128 MB escritos =
**256 MB, casi 3x el propio kernel de decodificacion (88,7 MB)**. Fusionarla
baja el trafico total del pipeline de ~345 MB a ~217 MB (−37%) y quita un
lanzamiento entero. Necesario para la Fase 3 de todos modos.

### 4.3 Indice de 16 bits para BLOCK >= 256

El de 8 bits no llega. La variante relativa de 16 bits da 2,016 B/bloque
(+1,94 puntos con BLOCK=256). Util si 4.1 confirma que BLOCK grande es viable.

### 4.4 Huffman sobre pares

Reduce a la mitad las iteraciones de la cadena serie, que es el cuello de
botella real, y ademas es 0,0164 bits/simbolo mas pequeno. Coste: LUT de
8 KiB. Especulativo pero barato de probar.

### 4.5 Comparar contra DFloat11

Esta publicado y trae kernels. Es la comparacion que cualquier revisor
exigiria, y es la que mas trabajo cuesta.

---

## 5. Trampas ya pisadas, para no repetirlas

- Un codigo sin Kraft <= 1 es indescifrable aunque parezca que funciona en
  algunos casos de prueba.
- Extrapolar un porcentaje de ahorro de un tipo de datos a otro no vale.
- Una tabla plana desperdicia el sesgo de la distribucion.
- **Un `return` temprano antes de `__syncthreads()` es comportamiento
  indefinido.** Costo salida incorrecta no determinista en la escalera con
  BLOCK=512/1024. Todos los hilos del bloque deben llegar a la barrera.
- **Verificar bit a bit ANTES de cronometrar, nunca despues.** Un kernel que
  escribe fuera de su shared puede dar resultados casi correctos y un tiempo
  halagador.
- **No combinar optimizaciones sin medirlas por separado.** Entrada + salida
  en shared es *peor* que solo salida: la entrada gasta shared y una barrera
  sin comprar nada.
- Los relojes de una GPU consumer no se pueden fijar bajo Windows/WDDM. Hay
  que compensar con calentamiento sostenido, mediana y orden aleatorizado — y
  cerrar todo lo que use la GPU.
- Nsight Compute no soporta Pascal (retirado en 2020.1). No se puede perfilar
  una 1050 Ti con ninguna version actual.

---

## 6. Estado del arte

**DFloat11** (NeurIPS 2025, arXiv 2504.11651,
github.com/LeanModels/DFloat11) hace lo mismo: Huffman sobre los exponentes
BF16, signo y mantisa intactos, ~30% de reduccion bit a bit identica. Usa LUT
jerarquicas en SRAM, kernel de dos fases y un array de *gaps* con el offset en
bits de cada hilo — equivalente al indice de bloques de aqui.

**Esto no es competencia con DFloat11.** Y su array de gaps tiene exactamente
la estructura que el indice de 8 bits mejora, asi que la via de contribucion
mas directa es abrir un issue o PR alli con esa medida.
