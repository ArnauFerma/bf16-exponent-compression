# Instrucciones para Cristian — prueba en la RTX 4070 (Windows 11)

Hola Cristian. Gracias por prestar la maquina.

Es un experimento de **compresion sin perdidas de pesos de modelos de IA**.
Hay dos formas de codificar los datos y hay que medir cual va mas rapida en
GPU. Tu tarjeta es justo la pieza que falta: la mia es una GTX 1050 Ti, de
2016, con 1 MB de cache L2 y demasiado antigua para las herramientas de
medida de NVIDIA.

**No hace falta saber nada del proyecto.** Son comandos de copiar y pegar.

- **Tiempo:** ~40 min, de los cuales ~30 son esperar.
- **Disco:** ~5 GB (se borra todo al final si quieres).
- **Todo queda en una carpeta**, salvo dos programas que puedes desinstalar.

---

## Por que precisamente tu tarjeta

Lo que se mide es si el rendimiento se hunde cuando los datos dejan de caber
en la cache L2 de la GPU. La cifra que manda no es el tamano de la L2 sino
**la L2 por hilo residente**:

| | L2 | SMs | hilos residentes | **L2 / hilo** |
|---|---|---|---|---|
| GTX 1050 Ti (la mia) | 1 MB | 6 | 12.288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43.008 | 73 B |
| RTX 4090 | 72 MB | 128 | 196.608 | 384 B |
| **RTX 4070 (la tuya)** | **36 MB** | 46 | 70.656 | **534 B** |

La 4070 tiene **mas L2 por hilo que una 4090**, porque tiene proporcionalmente
mas cache por SM. Para esta pregunta concreta es la mejor tarjeta de la lista,
6,3 veces mejor que la mia. No es un premio de consolacion: es literalmente el
mejor dato que puedo conseguir.

> ### Prediccion, escrita antes de medir
> En la 4070 el hundimiento al subir el parametro `BLOCK` **deberia
> desaparecer** hasta BLOCK=1024. Con 1024, los datos residentes ocupan
> 24,5 MB y tu L2 son 36 MB: **caben**. En mi tarjeta no caben ni con
> BLOCK=256, y por eso ahi se hunde.
>
> Si en la tuya tampoco se hunde, la explicacion queda confirmada. Si se hunde
> igual, mi explicacion es falsa — que tambien es informacion util.

---

## Que se instala (y como borrarlo)

| Que | Para que | Como se quita |
|---|---|---|
| Python 3.12 | ejecutar el programa | Configuracion > Aplicaciones |
| NVIDIA Nsight Compute | herramienta oficial de NVIDIA para medir | Configuracion > Aplicaciones |
| La carpeta del proyecto | codigo + un modelo de IA de 1,4 GB | borrar la carpeta |

Nada arranca con Windows, nada corre en segundo plano, no toca drivers ni
juegos.

---

## Paso 1 — Instalar Python y Nsight Compute

Abre **PowerShell** (tecla Windows, escribe `powershell`, Enter) y pega:

```powershell
winget install Python.Python.3.12
winget install Nvidia.Nsight.Compute
```

Acepta lo que pregunte. La segunda tarda unos minutos.

**Cierra PowerShell al acabar.** Si no, no encontrara Python despues.

## Paso 2 — Permitir la lectura de contadores de la GPU

Windows bloquea por defecto los contadores de rendimiento. Sin esto la parte
de medicion detallada falla (el resto funciona igual).

1. Clic derecho en el escritorio > **Panel de control de NVIDIA**
2. Menu de arriba: **Escritorio**
3. Marca **"Habilitar contadores de rendimiento de la GPU"**
   (o *Manage GPU Performance Counters* > **permitir a todos los usuarios**)
4. **Reinicia el PC**

> Si no encuentras la opcion, saltatelo. Mas adelante puedes abrir PowerShell
> **como administrador** (clic derecho > *Ejecutar como administrador*) y
> funciona igual.

## Paso 3 — Descomprimir y preparar

1. Copia `portable_bundle.zip` a una carpeta con **5 GB libres**, p. ej.
   `C:\prueba\`
2. Clic derecho en el zip > **Extraer todo**
3. Entra en la carpeta extraida
4. Clic derecho **dentro** de la carpeta, en un sitio vacio >
   **Abrir en Terminal**
5. Pega:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

Instala lo necesario, **descarga un modelo de 1,4 GB** (la parte lenta) y
comprueba que todo funciona.

### Como saber que ha ido bien

Tiene que aparecer al final:

```
RESULTADO: ambos kernels correctos
```

> **Si sale `FALLO` o un error en rojo, PARA AQUI** y mandame una captura. Sin
> esa comprobacion los tiempos no valen nada: un descompresor rapido que
> descomprime mal no sirve para nada.

## Paso 4 — Lanzar todas las medidas

**Cierra antes todo lo que use la GPU**: navegadores (Chrome, Edge), Discord,
Steam, Spotify, YouTube, juegos, Blender, OBS. Cuanto mas limpio, mas fiables
los numeros. En Windows no se pueden fijar las frecuencias de la GPU, asi que
esto es lo unico que hay.

```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

Te pedira confirmar que has cerrado todo y luego tarda **~25 minutos**. Va
imprimiendo 5 bloques:

1. barrido base (las dos formas de codificar, kernel original)
2. variantes del patron de acceso a memoria
3. las dos formas otra vez, con la version optimizada
4. indice de 8 bits
5. perfilado con Nsight Compute

Entre bloque y bloque puede pasar un rato sin imprimir nada. Es normal.
**No uses el PC mientras tanto**, falsea las medidas.

Al terminar deja un fichero **`resultados.zip`**.

## Paso 5 — Mandarmelo

Solo ese fichero:

```
resultados.zip
```

Son unos pocos KB de texto. **No me mandes** el modelo ni la carpeta `.venv\`:
pesan gigas y no sirven.

---

## Si algo falla

| Lo que ves | Que hacer |
|---|---|
| `python no se reconoce...` | no cerraste PowerShell tras el paso 1 |
| `ERR_NVGPUCTRPERM` | contadores: paso 2, o PowerShell como administrador |
| `ncu.exe no encontrado` | no se instalo Nsight Compute; el resto sigue valiendo |
| `Failed to find CUDA headers` | `.\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"` |
| `Windows error 32` en un DLL | espera 30 s y repite el comando; era una descarga aun abierta |
| `excede 48 KiB` en algunas filas | **es normal y esperado**, no es un error |
| Parece colgado | paso 3 descarga 1,4 GB; paso 4 tarda ~25 min |
| Cualquier otra cosa | para y mandame el texto. No se puede romper nada. |

## Para borrarlo todo

1. Borra la carpeta (`C:\prueba\...`)
2. Configuracion > Aplicaciones > desinstalar **Python 3.12** y
   **NVIDIA Nsight Compute**, si no los quieres conservar

Gracias de verdad. Tu tarjeta contesta una pregunta que en la mia es
fisicamente imposible de contestar.
