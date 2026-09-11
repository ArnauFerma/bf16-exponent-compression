# Prueba de kernels CUDA en la RTX 4070 — paso a paso

Hola David. Gracias por prestar la maquina.

Es un experimento de **compresion sin perdidas de pesos de modelos de IA**.
Hay dos formas de codificar y hay que medir cual va mas rapida en GPU. Tu
tarjeta es la pieza que falta: la mia es una GTX 1050 Ti (Pascal), demasiado
antigua para las herramientas de medida de NVIDIA y con solo 1 MB de L2.

**No hace falta saber nada del proyecto.** Son comandos de copiar y pegar.

- **Tiempo:** ~30 min, de los cuales ~20 son esperar.
- **Disco:** ~5 GB (se borra todo al final si quieres).
- **Todo queda dentro de la carpeta**, salvo dos paquetes de sistema
  (`python3-venv` y opcionalmente `nsight-compute`).

---

## Por que precisamente tu tarjeta

Lo que se esta probando es si el rendimiento se hunde cuando los datos
dejan de caber en la cache L2 de la GPU. La cifra que manda no es el
tamano de la L2 sino **la L2 por hilo residente**:

| | L2 | SMs | hilos residentes | **L2 / hilo** |
|---|---|---|---|---|
| GTX 1050 Ti (la mia) | 1 MB | 6 | 12.288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43.008 | 73 B |
| RTX 4090 | 72 MB | 128 | 196.608 | 384 B |
| **RTX 4070 (la tuya)** | **36 MB** | 46 | 70.656 | **534 B** |

La 4070 tiene mas L2 por hilo que una 4090, porque tiene proporcionalmente
mas cache por SM. Para esta pregunta concreta es **la mejor tarjeta de la
lista**, 6,3x mejor que la mia.

> ### Prediccion, escrita antes de medir
> En la 4070 el hundimiento de rendimiento al subir el parametro `BLOCK`
> **deberia desaparecer** hasta BLOCK=1024. Con 1024, los datos residentes
> ocupan 24,5 MB y la L2 son 36 MB: **caben**. En mi tarjeta no caben ni con
> BLOCK=256, y por eso ahi se hunde.
>
> Si en tu tarjeta tampoco se hunde, la explicacion queda confirmada. Si se
> hunde igual, mi explicacion es falsa — que tambien es informacion util.

---

## Paso 1 — Comprobar que la GPU responde

```bash
nvidia-smi
```

Tiene que salir una tabla con `NVIDIA GeForce RTX 4070`. Si dice
`command not found` o da error, faltan los drivers y ahi paro yo: avisame.

Mira tambien si la columna de procesos esta vacia. Tus dos monitores
deberian ir por la grafica integrada del Ryzen; si la 4070 no pinta
escritorio, las medidas saldran mucho mas limpias.

## Paso 2 — Instalar Nsight Compute (opcional pero muy util)

Es la herramienta oficial de NVIDIA para medir por dentro. Sin ella el
experimento funciona igual, pero se pierde la mitad de la informacion.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y nsight-compute
```

Comprueba con `ncu --version`. Si el binario no queda en el PATH:

```bash
export PATH=$PATH:/opt/nvidia/nsight-compute/$(ls /opt/nvidia/nsight-compute | tail -1)
```

### Permisos de contadores

Por defecto el driver solo deja leer los contadores a root. Dos opciones:

- **Facil:** no hacer nada. El script detecta que no hay permisos y usa
  `sudo` automaticamente para esa parte.
- **Permanente:**
  ```bash
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u
  ```
  y reiniciar.

## Paso 3 — Descomprimir y preparar

```bash
mkdir -p ~/bf16 && cd ~/bf16
tar xzf ~/Descargas/portable_bundle.tar.gz    # o unzip portable_bundle.zip
bash setup_linux.sh
```

Esto crea un entorno Python aislado (`.venv/`, no toca tu Python del
sistema), instala CuPy, **descarga un modelo de 1,4 GB** (la parte lenta) y
comprueba que todo funciona.

### Como saber que ha ido bien

Al final tiene que aparecer:

```
RESULTADO: ambos kernels correctos
```

> **Si sale `FALLO` o cualquier error, PARA AQUI** y mandame la salida. Sin
> esa comprobacion los tiempos no valen nada: un kernel rapido que
> descomprime mal no sirve para nada.

## Paso 4 — Lanzar todas las medidas

**Cierra antes lo que use la GPU** (navegadores con video, juegos, Blender,
cualquier cosa con CUDA). Si tus monitores van por la integrada del Ryzen,
con eso basta.

```bash
bash run_all.sh
```

Tarda ~20 minutos y va imprimiendo 4 bloques:

1. barrido base (Huffman vs escalera, kernel original)
2. variantes del patron de acceso
3. Huffman vs escalera con la optimizacion
4. perfilado con Nsight Compute

El script intenta ademas **fijar los relojes de la GPU** con `nvidia-smi
-lgc` (te pedira sudo). Eso quita el ruido de frecuencias variables; en
Windows no se puede y por eso mis medidas son mas sucias que las tuyas.
Si falla, no pasa nada, sigue igual.

Al terminar deja un fichero **`resultados.tar.gz`**.

## Paso 5 — Mandarme el resultado

Solo el fichero:

```
~/bf16/resultados.tar.gz
```

Son unos pocos KB de texto. **No me mandes** el modelo descargado ni la
carpeta `.venv/`: eso pesa gigas y no sirve.

---

## Si algo falla

| Lo que ves | Que hacer |
|---|---|
| `nvidia-smi: command not found` | faltan drivers, avisame |
| `ensurepip is not available` | `sudo apt install python3-venv python3-dev` |
| `Failed to find CUDA headers` | `./.venv/bin/python -m pip install "cupy-cuda12x[ctk]"` |
| `ERR_NVGPUCTRPERM` | permisos de contadores, paso 2 |
| `ncu: command not found` | no se instalo o no esta en el PATH, paso 2 |
| `cudaErrorIllegalAddress` | manda la salida entera, es un bug mio |
| Parece colgado | paso 3 descarga 1,4 GB y el paso 4 tarda ~20 min sin decir mucho |

Nada de esto puede tocar el sistema fuera de la carpeta.

## Para borrarlo todo

```bash
rm -rf ~/bf16
sudo apt remove nsight-compute      # solo si no lo quieres conservar
```

Gracias. Con esto se contesta una pregunta que en mi maquina es fisicamente
imposible de contestar.
