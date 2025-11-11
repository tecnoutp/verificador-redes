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

    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        if os.path.exists("Informe de errores.txt"):
            os.remove("Informe de errores.txt")

        # ejecutar tu programa
        Verificar(DatosT, DatosL, DatosN, CurTemp)

        with open("Informe de errores.txt", "rb") as f:
            st.download_button(
                "Descargar Informe",
                f,
                file_name="Informe_de_errores.txt",
                mime="text/plain"
            )
