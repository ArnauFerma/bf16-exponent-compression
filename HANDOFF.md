# Compresión lossless de pesos BF16 con código por niveles

Handoff para continuar en Claude Code / entorno con GPU.

---

## 0. Resumen en una frase

Un código de prefijo por niveles ("escalera") sobre el campo exponente de
BF16 consigue **31,28% de reducción lossless** frente al **32,08% de Huffman
canónico**, usando una tabla de 64 bytes en lugar de una LUT jerárquica en
SRAM. Falta medir si esa tabla pequeña compensa en rendimiento real.

---

## 1. Estado de las afirmaciones

Esto importa más que nada: separa lo medido de lo especulado.

### Medido y reproducible

| Afirmación | Valor | Cómo se verificó |
|---|---|---|
| Entropía del exponente BF16 | 2,718 bits | Conteo directo sobre 2,16M pesos |
| Huffman canónico | 2,758 bits/exp → 32,08% | Codec implementado, roundtrip bit a bit OK |
| Escalera propuesta | 2,870 bits/exp → 31,28% | Coste calculado sobre la distribución real |
| Kraft de la escalera | exactamente 1,0000 | Suma de 2^-longitud |
| Índice de bloques (256 símbolos, uint32) | 1,15% del total | Aritmética directa |
| Esquema plano (tabla de 8, todos igual longitud) | 23,56% | Mismo método |

### No medido — hipótesis a validar

| Hipótesis | Por qué es dudosa |
|---|---|
| La escalera decodifica más rápido que Huffman | Huffman canónico usa LUT: **un** acceso a SRAM. La escalera usa `clz` + rama + shift + mask. Contando instrucciones puede perder. |
| La tabla de 64 B mejora la ocupación | Plausible, pero la LUT de Huffman ya está diseñada para caber en SRAM. La ganancia puede ser nula. |
| El cuello de botella es el decode | Casi seguro que **no**. Manda el ancho de banda y la dependencia serie dentro del bloque. |

### Descartado (callejones sin salida ya recorridos)

- **Comprimir el flujo de bits crudo en grupos de 3 o 4 bits.** Sobre los
  pesos BF16 da entre −18% (expande) y +12,6% según configuración. Muy por
  debajo de atacar el exponente.
- **Índice de offsets por peso.** Cuesta ~800% del fichero. Inviable.
- **Prefix-sum global para calcular offsets.** Problema huevo-gallina: para
  sumar longitudes necesitas saber dónde empieza cada símbolo. La solución
  es el índice grueso por bloques.
- **Códigos no libres de prefijo.** Un intento previo usaba `100` y `1000`
  a la vez. Indescifrable. Cualquier tabla nueva **debe** pasar Kraft ≤ 1.

---

## 2. Por qué el exponente

BF16 = `[1 signo][8 exponente][7 mantisa]`.

Entropías medidas sobre pesos sintéticos realistas:

```
signo     : 1,000 bits de 1   → nada que ganar
exponente : 2,718 bits de 8   → 5,3 bits desperdiciados por peso
mantisa   : 6,972 bits de 7   → nada que ganar
```

Solo aparecen **25 exponentes distintos** de 256 posibles. Tres de ellos
concentran el 69,7% de los pesos. Ahí está todo el margen.

Signo y mantisa se guardan crudos, empaquetados en 8 bits por peso en un
array separado. No se tocan.

---

## 3. Especificación del formato

### 3.1 Código por niveles

Lee unos hasta el primer cero. El número de unos selecciona el nivel.

| Código | Bits | Contenido |
|---|---|---|
| `0` + 1 bit índice | 2 | símbolos 0–1 (los 2 más frecuentes) |
| `10` + 1 bit índice | 3 | símbolos 2–3 |
| `110` + 1 bit índice | 4 | símbolos 4–5 |
| `1110` + 2 bits índice | 6 | símbolos 6–9 |
| `1111` + 8 bits crudos | 12 | escape, cualquier otro exponente |

Kraft: `2·2⁻² + 2·2⁻³ + 2·2⁻⁴ + 4·2⁻⁶ + 2⁻⁴ = 1,0000` exacto.

Tabla concreta para el fichero de pruebas:

| Código | Bits | Exponente | % |
|---|---|---|---|
| `00` | 2 | 120 | 28,10 |
| `01` | 2 | 121 | 21,73 |
| `100` | 3 | 119 | 19,84 |
| `101` | 3 | 118 | 10,89 |
| `1100` | 4 | 122 | 7,54 |
| `1101` | 4 | 117 | 5,59 |
| `111000` | 6 | 116 | 2,80 |
| `111001` | 6 | 115 | 1,41 |
| `111010` | 6 | 114 | 0,69 |
| `111011` | 6 | 123 | 0,61 |
| `1111`+8b | 12 | resto | 0,79 |

La **forma** de la escalera es fija. Lo único que cambia entre modelos es
qué exponente ocupa cada ranura: 10 bytes de cabecera.

### 3.2 Decodificación

```c
// n = número de unos iniciales
int n = clz(~(word << pos));   // una instrucción
switch (n) {
  case 0: sym = tabla[ 0 + bit(pos+1)      ]; len = 2;  break;
  case 1: sym = tabla[ 2 + bit(pos+2)      ]; len = 3;  break;
  case 2: sym = tabla[ 4 + bit(pos+3)      ]; len = 4;  break;
  case 3: sym = tabla[ 6 + bits(pos+4, 2)  ]; len = 6;  break;
  default: sym = bits(pos+4, 8);              len = 12; break;
}
```

Sin árbol, sin bucle. La tabla son 10 bytes de símbolos.

### 3.3 Layout del fichero

```
cabecera        : magic, versión, n_pesos, BLOCK, 10 bytes de tabla
sign_mantisa    : n_pesos bytes, crudos
exponentes      : bitstream del código por niveles
índice          : uint32 por bloque de BLOCK símbolos → offset en bits
```

Tamaños sobre el fichero de pruebas (2.162.688 pesos, 4.325.376 B):

```
exponentes      :   775.869 B
signo+mantisa   : 2.162.688 B
índice          :    33.792 B   (1,15%)
tabla           :        64 B
-----------------------------------
TOTAL           : 2.972.325 B   →  31,28% de ahorro
```

### 3.4 Paralelismo

El índice por bloques resuelve el huevo-gallina. Cada bloque de 256
símbolos arranca en un offset conocido, así que **8.448 bloques se
decodifican en paralelo** sin sincronización. Dentro del bloque es serie,
pero son 256 símbolos.

`BLOCK` es el parámetro a barrer: más grande → índice más pequeño pero
menos paralelismo y cadenas serie más largas.

---

## 4. Ficheros

| Fichero | Qué es |
|---|---|
| `weights_bf16.bin` | 2.162.688 pesos BF16, 4,3 MB. Banco de pruebas. |
| `gen_weights_bf16.py` | Genera el anterior. Semilla 1234, reproducible. |
| `df11_reference.py` | Huffman canónico + índice de bloques. Baseline a batir. Roundtrip verificado. |

`weights_bf16.bin` es **sintético**: 7 tensores gaussianos con escalas de
0,009 a 0,030, más 0,3% de outliers a 8σ. Reproduce la entropía de
exponente de los modelos reales (2,72 vs ~2,6 reportado), que es la
propiedad que importa.

**Primera tarea en el entorno nuevo: repetir todo sobre pesos reales.**
Un `.safetensors` de cualquier modelo pequeño (Qwen3-0.6B, Llama-3.2-1B).
Si la distribución del exponente cambia mucho, la escalera hay que
recalcularla.

---

## 5. Plan de pruebas

### Fase 1 — Validación en CPU

1. Descargar un modelo real, extraer un tensor grande.
2. Medir entropía del exponente. **Criterio: entre 2,4 y 3,0 bits.**
3. Correr `df11_reference.py`. **Criterio: ≥28% de ahorro, roundtrip exacto.**
4. Recalcular la asignación de la escalera para esa distribución.
5. Implementar el codec de escalera. **Criterio: roundtrip exacto y a menos
   de 1 punto de Huffman.**

Si el paso 2 falla, la premisa se cae y hay que parar a repensar.

### Fase 2 — Kernel GPU

Dos kernels con interfaz idéntica: Huffman-LUT y escalera. Mismo índice de
bloques, mismo layout, misma salida.

Medir con Nsight Compute:

- ciclos por símbolo decodificado
- ocupación (warps activos / máximo)
- bytes de SRAM por bloque
- ancho de banda alcanzado vs pico

**La pregunta que decide todo:** ¿el kernel está limitado por memoria o por
cómputo? Si es por memoria, la escalera no aporta velocidad y su único
argumento es el tamaño de tabla. Si es por cómputo, hay carrera.

### Fase 3 — Integración

Descomprimir antes del matmul, descartar después. Medir tokens/s con batch
1, 8, 32 frente a BF16 sin comprimir.

Referencia publicada: DFloat11 reporta ~2× más lento con batch 1, y la
penalización se diluye al subir el batch porque el coste de descompresión
es constante por forward pass.

### Barridos

- `BLOCK` ∈ {64, 128, 256, 512, 1024}
- forma de la escalera: probar (1,1,1,2) contra (1,1,2,2) y (1,2,2,3)
- ancho del índice: uint32 vs uint16 con offsets relativos al bloque

---

## 6. Riesgos

**El más probable:** que la escalera no gane nada en velocidad y quede como
una curiosidad 0,8% peor que Huffman. Sería un resultado válido y hay que
estar dispuesto a aceptarlo.

**Segundo:** que los pesos reales tengan una distribución de exponente
distinta a la sintética. Se descubre en la fase 1, barato.

**Tercero:** que el coste de descompresión por forward pass haga el
conjunto inviable fuera de escenarios con memoria muy justa. Es lo que ya
le pasa a DFloat11 con batch pequeño.

**Cuarto:** la rama del escape (`1111`) provoca divergencia de warp. Solo
el 0,79% de los símbolos, pero en GPU un solo hilo divergente serializa el
warp entero. Medirlo, no asumirlo.

---

## 7. Estado del arte

**DFloat11** (NeurIPS 2025, arXiv 2504.11651, github.com/LeanModels/DFloat11)
hace exactamente esto: Huffman sobre los exponentes BF16, signo y mantisa
intactos, ~30% de reducción con salidas bit a bit idénticas.

Su kernel usa LUT jerárquicas en SRAM, un kernel de dos fases con variables
auxiliares para coordinar lectura y escritura por hilo, y descompresión por
bloque de transformer. Guardan un array de *gaps* con el offset en bits del
primer elemento de cada hilo — equivalente al índice de bloques de aquí.

**Esto no es competencia con DFloat11.** Es una variante del código de
entropía dentro del mismo marco. La pregunta acotada es si un código por
niveles con tabla de 64 bytes rinde mejor que una LUT jerárquica, aceptando
0,8 puntos peor de compresión.

Comparar contra su implementación directamente. Está publicada.

---

## 8. Contexto de cómo se llegó aquí

El diseño salió de razonar desde cero sobre teoría de la información:
por qué los códigos de longitud variable rompen el paralelismo, por qué un
índice por elemento no puede funcionar, por qué un índice grueso sí. Se
reinventaron y descartaron RLE, coincidencia de bloques 2D y compresión vía
codecs de imagen por hardware antes de llegar aquí.

Las trampas ya pisadas, para no repetirlas:

- Un código sin Kraft ≤ 1 es indescifrable, aunque parezca que funciona en
  algunos casos de prueba.
- Los patrones frecuentes en texto ASCII no dicen nada sobre pesos.
- Extrapolar un porcentaje de ahorro de un tipo de datos a otro no vale.
- Una tabla plana desperdicia el sesgo: si un símbolo sale el 28% y otro el
  10%, darles la misma longitud tira compresión a la basura.
