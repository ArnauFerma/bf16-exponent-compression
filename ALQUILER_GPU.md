# Alquilar una GPU por horas para las medidas que faltan

Documento autocontenido: con esto se puede hacer la prueba sin consultar nada
mas. Coste total **~1,40 USD**, tiempo real **menos de una hora**.

---

## 1. Que falta medir y por que hace falta otra tarjeta

Todo lo medido hasta ahora es una GTX 1050 Ti: Pascal, 6 SMs, **1 MiB de L2**.
Dos limitaciones que no se pueden sortear en esa maquina:

1. El rendimiento se hunde al subir el parametro `BLOCK` (9,7 ms con BLOCK=64,
   154 ms con BLOCK=1024). La explicacion propuesta es que el working set
   residente desborda la L2. **En una tarjeta con L2 grande eso deberia
   desaparecer** — y BLOCK grande es justo donde mejor comprime el formato.
2. Nsight Compute **retiro el soporte de Pascal** en 2020.1. No hay forma de
   perfilar esa tarjeta con ninguna version actual.

## 2. Que tarjeta alquilar

La cifra que gobierna el efecto **no** es el tamano de la L2 ni la velocidad
bruta, sino la **L2 por hilo residente**:

| tarjeta | L2 | SMs | hilos residentes | **L2 / hilo** |
|---|---|---|---|---|
| GTX 1050 Ti (la de casa) | 1 MB | 6 | 12.288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43.008 | 73 B |
| **A100 80GB** | 40 MB | 108 | 221.184 | **190 B** |
| H100 SXM | 50 MB | 132 | 270.336 | 194 B |
| RTX 4090 | 72 MB | 128 | 196.608 | 384 B |
| RTX 4070 | 36 MB | 46 | 70.656 | 534 B |

### Recomendacion: **A100 80GB, ~1,39 USD/hora**

Dos razones:

1. **Rellena el hueco del medio.** Las tarjetas disponibles gratis se agolpan
   en los extremos (85 B y 534 B). Con la A100 quedan cuatro puntos repartidos:
   85 -> 190 -> 384 -> 534.
2. **Es el hardware objetivo de DFloat11.** Cualquier afirmacion pasa a ser
   directamente comparable con sus numeros publicados, en vez de una discusion
   sobre tarjetas de consumo.

### Alternativa barata: **RTX 4090, ~0,34 USD/hora**

Util, pero con 384 B/hilo queda cerca de una 4070, asi que aporta menos
informacion por euro. Elegirla solo si se quiere gastar centimos.

### No alquilar

**RTX 3090 ni RTX A6000**: solo 6 MB de L2 y muchos hilos residentes, o sea
~50 B/hilo — **peor que la 1050 Ti**. Nada anterior a Turing.

## 3. Donde

| proveedor | RTX 4090 | A100 80GB | notas |
|---|---|---|---|
| **RunPod** Community | ~0,34 $/h | ~1,39 $/h | interfaz sencilla, cobro por segundo, sin coste de trafico |
| RunPod Secure | ~0,69 $/h | ~1,39 $/h | con SLA; innecesario aqui |
| **Vast.ai** on-demand | ~0,29–0,59 $/h | variable | lo mas barato, pero es peer-to-peer y la calidad varia |
| Vast.ai spot | ~0,11–0,35 $/h | variable | interrumpible; aceptable para 30 minutos |

Para una prueba puntual: **RunPod Community**. Se tarda menos en desplegar que
en comparar maquinas en Vast.

- RunPod: <https://www.runpod.io/pricing>
- Vast.ai: <https://vast.ai/pricing/gpu/RTX-4090>
- Comparador: <https://getdeploying.com/gpus/nvidia-rtx-4090>

## 4. Aviso importante: el perfilado puede no funcionar (y da igual)

Dentro de un contenedor alquilado, `ncu` suele fallar con `ERR_NVGPUCTRPERM`:
el permiso de contadores lo controla el **anfitrion**, no el contenedor, y no
se puede arreglar desde dentro.

**No pasa nada y no hay que buscar una instancia que lo permita.** Los
barridos de tiempo — que son los que responden la pregunta de la L2 — no
necesitan `ncu`, y `run_all.sh` lo detecta y se lo salta limpiamente. Coger la
instancia barata.

## 5. Paso a paso (RunPod)

1. Registrarse en <https://runpod.io> y anadir saldo (el minimo son 10 USD; se
   van a gastar ~1,40).
2. **Deploy** -> pestana **Community Cloud** -> elegir **A100 80GB**
   (o RTX 4090).
3. Plantilla: cualquier imagen de PyTorch o CUDA sobre Ubuntu.
   **Container disk: 20 GB como minimo** — el modelo mas los pesos extraidos
   ocupan ~4 GB y el valor por defecto a veces se queda corto.
4. Desplegar y esperar a que quede en *Running*. Luego **Connect** -> terminal
   web o **JupyterLab**.
5. Subir el bundle. Tres opciones, de mas facil a menos:
   - **JupyterLab**: abrirlo y **arrastrar `portable_bundle.tar.gz`** al
     explorador de ficheros de la izquierda.
   - **runpodctl** (si esta instalado en el PC de origen):
     ```bash
     runpodctl send portable_bundle.tar.gz
     ```
     y ejecutar en el pod el comando `receive` que imprime.
   - **Descargarlo dentro del pod**, si esta subido a algun sitio accesible.
6. En la terminal del pod:
   ```bash
   mkdir -p ~/bf16 && cd ~/bf16
   tar xzf ~/portable_bundle.tar.gz      # ajustar la ruta si hace falta
   bash setup_cloud.sh
   ```
   `setup_cloud.sh` es la variante pensada para contenedores: funciona siendo
   root, sin `sudo`, y con imagenes que no traen el modulo `venv`.

   **Comprobar dos cosas en su salida:**

   - la linea `L2 por hilo residente:` — en una A100 deben salir ~190 B. Si
     sale algo cercano a 50–85 B, la instancia no es la tarjeta que se pidio:
     terminarla y volver a elegir.
   - al final, `RESULTADO: ambos kernels correctos`. Si falla, el script sale
     con error a proposito: **no seguir**, los tiempos no valdrian nada.

7. Lanzar todo:
   ```bash
   bash run_all.sh
   ```
   ~20 minutos. Deja `resultados.tar.gz`.

8. Recuperar el fichero: descargarlo desde JupyterLab (clic derecho ->
   Download) o `runpodctl send resultados.tar.gz` desde el pod.

9. **TERMINAR el pod.** No "Stop": **Terminate**. Un pod parado sigue cobrando
   almacenamiento.

## 6. Coste

| | A100 80GB | RTX 4090 |
|---|---|---|
| preparacion + descarga del modelo | ~10 min | ~10 min |
| medidas | ~15 min | ~15 min |
| **total (~40 min, redondeado a 1 h)** | **~1,40 USD** | **~0,35 USD** |

## 7. Que mirar en el resultado

La prediccion, registrada antes de medir, para poder leer el resultado sin
ayuda:

> En una tarjeta con L2 grande el **precipicio de BLOCK deberia aplanarse**. En
> la 1050 Ti, pasar de BLOCK=64 a BLOCK=1024 cuesta ~14x en tiempo. Si la
> explicacion de la L2 es correcta, esa proporcion deberia encogerse mucho al
> subir la L2 por hilo, y la puesta de la **entrada** en shared deberia dejar
> de ayudar con BLOCK=256, porque los datos ya caben en cache.

Los ficheros a mirar dentro de `resultados.tar.gz`:

- `env_info.json` — versiones y tarjeta, sin datos de la maquina ni de la cuenta.
- `1_bench_base.txt` — el barrido de BLOCK. **Es el que contesta la pregunta.**
- `2_bench_opt.txt` — atribucion: si `smem in=1` deja de ganar con BLOCK=256,
  la explicacion de la L2 queda confirmada.
- `3_head2head.txt` — Huffman vs escalera con el kernel optimizado.
- `4_bench_idx8.txt` — indice de 8 bits.
- `ncu_out/` — solo si el perfilado funciono.

**Si el precipicio sigue ahi, la explicacion de la L2 es falsa** y hay que
reescribir la seccion 2b de RESULTS.md antes de sacar nada de esto del repo.

## 8. Si algo falla

| Lo que se ve | Que hacer |
|---|---|
| `nvidia-smi` no responde | la instancia no tiene GPU visible; terminar y coger otra |
| `L2 por hilo` no coincide con la tarjeta pedida | no es la tarjeta anunciada; terminar y coger otra |
| `FALLO` en la verificacion | parar; guardar la salida completa |
| `ERR_NVGPUCTRPERM` | perfilado bloqueado por el anfitrion; **ignorar**, el resto vale |
| `No space left on device` | el container disk se quedo corto; rehacer con 20+ GB |
| falta `venv` | `setup_cloud.sh` ya lo maneja solo |
| la instancia spot se corta a media prueba | relanzar `run_all.sh`; el modelo ya descargado se reutiliza |
