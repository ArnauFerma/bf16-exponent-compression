# Instrucciones para ejecutar la prueba en tu PC

Hola. Gracias por prestar la maquina.

Esto es un experimento de **compresion de modelos de IA**. Necesita una
grafica NVIDIA moderna y la tuya (RTX 3060) sirve; la mia no, porque es
demasiado antigua para las herramientas de medida de NVIDIA.

**No hay que saber nada del tema.** Son 5 pasos, copiar y pegar comandos.
Casi todo el tiempo es esperar descargas.

- **Tiempo total:** ~35 minutos, de los cuales ~25 son esperar.
- **Espacio en disco:** unos 5 GB (se puede borrar todo al acabar).
- **Un reinicio** en el paso 2.

---

## Que se instala (y como borrarlo despues)

Para que sepas exactamente que entra en tu PC:

| Que | Para que | Como se quita |
|---|---|---|
| Python 3.12 | ejecutar el programa | Configuracion > Aplicaciones > Desinstalar |
| NVIDIA Nsight Compute | herramienta oficial de NVIDIA para medir la grafica | Configuracion > Aplicaciones > Desinstalar |
| La carpeta del proyecto | codigo + un modelo de IA de 1,4 GB que se descarga | borrar la carpeta |

Nada se ejecuta en segundo plano, nada arranca con Windows, no toca tus
drivers ni tus juegos. Al borrar la carpeta y desinstalar esos dos programas
el PC queda exactamente como estaba.

---

## Paso 1 — Instalar Python y la herramienta de NVIDIA

Abre **PowerShell** (tecla Windows, escribe `powershell`, Enter) y pega
estas dos lineas, una detras de otra:

```powershell
winget install Python.Python.3.12
winget install Nvidia.Nsight.Compute
```

Si pregunta algo, acepta. La segunda tarda unos minutos.

**Cierra PowerShell al acabar** (importante: si no, no encuentra Python).

---

## Paso 2 — Dar permiso para medir la grafica

Windows bloquea por defecto los contadores de rendimiento de la grafica.
Sin esto la parte de medicion falla.

1. Clic derecho en el escritorio > **Panel de control de NVIDIA**
2. Menu de arriba: **Escritorio**
3. Marca **"Habilitar contadores de rendimiento de la GPU"**
   (o *Manage GPU Performance Counters* > **permitir a todos los usuarios**)
4. **Reinicia el PC**

> Si no encuentras esa opcion, no pasa nada: se puede saltar este paso y
> mas adelante abrir PowerShell **como administrador** (clic derecho >
> *Ejecutar como administrador*). Funciona igual.

---

## Paso 3 — Descomprimir y preparar

1. Copia `portable_bundle.zip` a una carpeta con **5 GB libres**, por
   ejemplo `C:\prueba\`
2. Clic derecho en el zip > **Extraer todo**
3. Entra en la carpeta extraida
4. Clic derecho **dentro** de la carpeta (en un sitio vacio) >
   **Abrir en Terminal** (o *Abrir ventana de PowerShell aqui*)
5. Pega esto:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

Esto instala lo necesario, **descarga un modelo de IA de 1,4 GB** (es la
parte lenta, ~10 min segun tu conexion) y comprueba que todo funciona.

### Como saber que ha ido bien

Al final tiene que salir esto:

```
RESULTADO: ambos kernels correctos
```

> **Si sale `FALLO` o cualquier error rojo, PARA AQUI** y mandame una foto
> o copia del texto. No sigas con los pasos siguientes: los resultados no
> valdrian para nada.

---

## Paso 4 — La medicion principal

**Antes de empezar, cierra todo lo que use la grafica:** navegadores
(Chrome, Edge), Discord, WhatsApp, Steam, Spotify, YouTube, juegos.
Cuanto mas limpio, mas fiables salen los numeros. Deja solo la ventana
de PowerShell.

Pega esto:

```powershell
.\.venv\Scripts\python.exe bench_gpu.py 64000000 > resultados_bench.txt
```

Tarda unos **5 minutos** y no muestra nada mientras trabaja (va escribiendo
al fichero). Es normal. **No toques el PC mientras tanto.**

---

## Paso 5 — La medicion detallada

```powershell
powershell -ExecutionPolicy Bypass -File profile_ncu.ps1
```

Tarda unos **10 minutos**. Va escribiendo lineas amarillas tipo
`-- BLOCK=256 codec=ladder`. Son 15 en total.

> Si aparece `ERR_NVGPUCTRPERM`, es el permiso del paso 2. Cierra
> PowerShell, abrelo **como administrador**, vuelve a la carpeta con
> `cd C:\prueba\...` y repite este paso.

---

## Paso 6 — Mandarme los resultados

Comprime y mandame **estas tres cosas** de la carpeta:

1. El fichero **`resultados_bench.txt`**
2. La carpeta entera **`ncu_out\`** (15 ficheros .csv)
3. El fichero **`outputs\gpu_bench.json`**

Son ficheros de texto, pesan poco (unos pocos KB). Por WhatsApp o correo
va sobrado.

**No hace falta que me mandes** el modelo descargado ni la carpeta
`.venv\` — eso pesa gigas y no sirve de nada.

---

## Si algo va mal

| Lo que ves | Que hacer |
|---|---|
| `python no se reconoce...` | No cerraste PowerShell tras el paso 1. Cierralo y abrelo otra vez. |
| `ERR_NVGPUCTRPERM` | Permisos: paso 2, o PowerShell como administrador. |
| `ncu.exe no encontrado` | No se instalo Nsight Compute. Repite `winget install Nvidia.Nsight.Compute`. |
| `Failed to find CUDA headers` | Ejecuta: `.\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"` |
| `Windows error 32` en un DLL | Espera 30 segundos y repite el mismo comando. Es una descarga que aun estaba abierta. |
| Se queda parado mucho rato | En el paso 3 es normal (descarga de 1,4 GB). En el 4 tambien (5 min sin mostrar nada). |
| Cualquier otra cosa | Para y mandame el texto del error. No pasa nada malo. |

Nada de esto puede romper el PC. En el peor caso borras la carpeta y ya.

---

## Para borrarlo todo al acabar

1. Borra la carpeta del proyecto (`C:\prueba\...`)
2. Configuracion > Aplicaciones > desinstalar **Python 3.12** y
   **NVIDIA Nsight Compute** (si no los quieres conservar)

Gracias de verdad. Con esto se resuelve una duda que en mi maquina no se
puede contestar.
