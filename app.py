import streamlit as st
import pandas as pd
import numpy as np
import io, os, tempfile
from Verificar import Verificar   # <<--- usa tu Verificar.py

st.title("Verificación de Circuitos Eléctricos")
st.write("Sube los 4 archivos CSV para validar el sistema.")

trafos_file   = st.file_uploader("Trafos.csv",   type=["csv"])
tramos_file   = st.file_uploader("Tramos.csv",   type=["csv"])
usuarios_file = st.file_uploader("Usuarios.csv", type=["csv"])
curvas_file   = st.file_uploader("Curvas.csv",   type=["csv"])

def _read_csv(f):
    b = f.read()
    return pd.read_csv(io.BytesIO(b))

if st.button("Ejecutar Verificación"):
    if not all([trafos_file, tramos_file, usuarios_file, curvas_file]):
        st.error("Debes subir los 4 archivos.")
        st.stop()

    dfT = _read_csv(trafos_file)
    dfL = _read_csv(tramos_file)
    dfN = _read_csv(usuarios_file)
    dfC = _read_csv(curvas_file)

    DatosT = dfT.select_dtypes(include=["number"]).to_numpy(dtype=float).ravel()
    DatosL = dfL.select_dtypes(include=["number"]).to_numpy(dtype=float)
    DatosN = dfN.select_dtypes(include=["number"]).to_numpy(dtype=float)
    CurTemp = dfC.select_dtypes(include=["number"]).to_numpy(dtype=float)

        # Ejecutar Verificar en un directorio temporal
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        outname = "Informe de errores.txt"
        try:
            if os.path.exists(outname):
                os.remove(outname)

            # Ejecutar tu verificador
            try:
                _err, _datos = Verificar(DatosT, DatosL, DatosN, CurTemp)
            except TypeError:
                # Por compatibilidad si Verificar devuelve 1 o 0 valores
                Verificar(DatosT, DatosL, DatosN, CurTemp)

            # Si no se creó el TXT, créalo con un mensaje mínimo
            if not os.path.exists(outname):
                with open(outname, "w", encoding="utf-8") as f:
                    f.write("No se generó 'Informe de errores.txt' durante la ejecución.\n")
                    f.write("Verifica que Verificar(DatosT, DatosL, DatosN, CurTemp) lo escriba en el directorio actual.\n")

        except Exception as e:
            # Captura cualquier error y lo vuelca al TXT para poder descargarlo
            st.error("Ocurrió un error en la ejecución. Se generará un informe con el stacktrace.")
            with open(outname, "w", encoding="utf-8") as f:
                f.write("ERROR DURANTE LA EJECUCIÓN DE VERIFICAR:\n\n")
                f.write(traceback.format_exc())
        finally:
            # Ofrecer descarga SIEMPRE
            if os.path.exists(outname):
                with open(outname, "rb") as f:
                    st.download_button(
                        "Descargar Informe",
                        f,
                        file_name="Informe_de_errores.txt",
                        mime="text/plain"
                    )
            else:
                st.error("No se pudo crear ni encontrar 'Informe de errores.txt'. Revisa tu función Verificar.")
            os.chdir(old_cwd)
