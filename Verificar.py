# Verificar.py
# Revisión eficiente con pandas/numpy/networkx.
# Cada bloque está comentado con el código de error y el método de verificación.

import numpy as np
import networkx as nx

# === Tablas de compatibilidad de faseos (tramo→tramo y tramo→usuario) ===
# Índices 1..7. Índice 0 no usado.
COMP_TT = np.zeros((8, 8), dtype=bool)   # compatible entre tramos consecutivos
COMP_TU = np.zeros((8, 8), dtype=bool)   # compatible tramo→fase_usuario

def _allow_TT(a, bs):
    for b in bs:
        COMP_TT[a, b] = True

def _allow_TU(t, us):
    for u in us:
        COMP_TU[t, u] = True

# Reglas (idénticas a tu lógica original)
_allow_TT(1, [1, 4, 6, 7])
_allow_TT(2, [2, 4, 5, 7])
_allow_TT(3, [3, 5, 6, 7])
_allow_TT(4, [1, 2, 4, 7])
_allow_TT(5, [2, 3, 5, 7])
_allow_TT(6, [1, 3, 6, 7])
_allow_TT(7, [1, 2, 3, 4, 5, 6, 7])  # si aplica cualquiera (dejar así para permitir 7 con todos)

_allow_TU(1, [1])
_allow_TU(2, [2])
_allow_TU(3, [3])
_allow_TU(4, [1, 2, 4])
_allow_TU(5, [2, 3, 5])
_allow_TU(6, [1, 3, 6])
_allow_TU(7, [1, 2, 3, 4, 5, 6, 7])  # si aplica cualquiera

def _write_log(lines):
    # Helper de I/O: acumula y escribe una vez
    if not lines:
        return
    with open('Informe de errores.txt', 'a', encoding='utf-8') as fid:
        for s in lines:
            fid.write(s)

def Verificar(DatosT: np.ndarray, DatosL: np.ndarray, DatosN: np.ndarray, CurTemp: np.ndarray):
    """
    Revisa coherencia de entrada y topología eléctrica, devolviendo:
        Error (int) y (posible) DatosT actualizado (se respeta tu firma).
    Escribe un informe en 'Informe de errores.txt'.
    """

    log = []
    circ = int(DatosT[0]) if DatosT.size and not np.isnan(DatosT[0]) else -1

    # =======================
    # (35, 36, 37) Presencia de datos y curva de carga
    # =======================
    
    # 35 -> No hay info de trafos ni usuarios
    if (DatosL.size == 0) and (DatosN.size == 0):
        _write_log([f"\r\nCircuito: {circ}\r\nError: 35\r\nNo hay información de trafos ni de usuarios\r\n"])
        return 35, DatosT

    # 36 -> Curva de carga está en ceros todas las horas
    if CurTemp.size and np.nansum(CurTemp) == 0:
        _write_log([f"\r\nCircuito: {circ}\r\nError: 36\r\nLa curva de carga está en ceros\r\n"])
        return 36, DatosT

    # 37 -> No tiene curva de carga
    if CurTemp.size == 0:
        _write_log([f"\r\nCircuito: {circ}\r\nError: 37\r\nNo tiene curva de carga\r\n"])
        return 37, DatosT

    # =======================
    # (2, 3, 4, 1) Coherencia básica de DatosT y slack
    # =======================
    
    tipo = int(DatosT[2]) if DatosT.size > 2 and not np.isnan(DatosT[2]) else -999
    topo = int(DatosT[5]) if DatosT.size > 5 and not np.isnan(DatosT[5]) else -999  # 1 radial / 0 enmallado
    vp = DatosT[3] if DatosT.size > 3 else np.nan
    vs = DatosT[4] if DatosT.size > 4 else np.nan
    slack = int(DatosT[0]) if DatosT.size and not np.isnan(DatosT[0]) else -1

    # 2 -> tipo trafo desconocido (debe ser 1 o 3)
    if tipo not in (1, 3):
        _write_log([f"\r\nCircuito: {circ}\r\nError: 2\r\nSe desconoce el tipo de transformador (1 o 3 - Monofásico o Trifásico)\r\n"])
        return 2, DatosT

    # 3 -> vp <= vs
    if not (np.isfinite(vp) and np.isfinite(vs)) or (vp <= vs):
        _write_log([f"\r\nCircuito: {circ}\r\nError: 3\r\nEl voltaje en el primario es menor o igual al voltaje del secundario\r\n"])
        return 3, DatosT

    # 4 -> topología desconocida (debe ser 0 o 1)
    if topo not in (0, 1):
        _write_log([f"\r\nCircuito: {circ}\r\nError: 4\r\nSe desconoce la topología del circuito (1 o 0 - Radial o Enmallado)\r\n"])
        return 4, DatosT

    # 1 -> slack no aparece en tramos (si hay tramos)
    if (DatosL.size != 0) and (slack != -1):
        ni_nf = DatosL[:, :2].astype(int) if DatosL.size else np.empty((0, 2), int)
        if ni_nf.size and (not np.any(ni_nf == slack)):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 1\r\nEl nodo del transformador (slack) no aparece en la hoja de tramos\r\n"])
            return 1, DatosT

    # =======================
    # (8, 9, 11, 13, 29) Checks rápidos y vectorizados sobre tramos
    # =======================
    
    if DatosL.size:
        fase_tramo = DatosL[:, 2].astype(int)
        montaje = DatosL[:, 4].astype(int)
        matF = DatosL[:, 6].astype(int)
        matN = DatosL[:, 8].astype(int)
        e = DatosL[:, :2].astype(int)

        # 8 -> Faseos de tramos no permitidos por tipo de trafo
        ok_mono = np.isin(fase_tramo, [1, 2, 4])
        ok_tri  = np.isin(fase_tramo, [1, 2, 3, 4, 5, 6, 7])
        if (tipo == 1 and not np.all(ok_mono)) or (tipo == 3 and not np.all(ok_tri)):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 8\r\nHay faseos en tramos que no corresponden al tipo de transformador\r\n"])
            return 8, DatosT

        # 9 -> Montaje ∈ {1,2}
        if not np.all(np.isin(montaje, [1, 2])):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 9\r\nExisten montajes desconocidos en tramos (1 o 2 - Abierta o Junta)\r\n"])
            return 9, DatosT

        # 11 -> Material fases ∈ {1,2}
        if not np.all(np.isin(matF, [1, 2])):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 11\r\nMaterial de fase desconocido (1 o 2 - Cobre o Aluminio)\r\n"])
            return 11, DatosT

        # 13 -> Material neutro ∈ {1,2}
        if not np.all(np.isin(matN, [1, 2])):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 13\r\nMaterial de neutro desconocido (1 o 2 - Cobre o Aluminio)\r\n"])
            return 13, DatosT

        # 29 -> Tramos con Ni == Nf (lazos)
        self_loops = np.where(e[:, 0] == e[:, 1])[0]
        if self_loops.size:
            lines = [f"\r\nCircuito: {circ}\r\nError: 29\r\nEn los siguientes tramos Ni == Nf\r\n"]
            for idx in self_loops:
                lines.append(f"{int(e[idx,0])}  {int(e[idx,1])}\r\n")
            _write_log(lines)
            return 29, DatosT

    # =======================
    # (14, 15, 33, 34, 16, 18, 19, 20, 27) Checks sobre usuarios
    # =======================
    
    if DatosN.size:
        nod_u = DatosN[:, 0].astype(int)
        fase_u = DatosN[:, 1].astype(int)
        med = DatosN[:, 4].astype(int)
        est = DatosN[:, 5].astype(int)
        clas = DatosN[:, 6].astype(int)

        # 14 -> Usuarios en nodos que no están en tramos (si hay tramos)
        if DatosL.size:
            nod_linea = np.unique(DatosL[:, :2].astype(int).ravel())
            faltan = [n for n in np.unique(nod_u) if n not in nod_linea]
            if len(faltan):
                lines = [f"\r\nCircuito: {circ}\r\nError: 14\r\nUsuarios en nodos que no aparecen en tramos:\r\n"]
                for u in faltan:
                    lines.append(f"{u}\r\n")
                _write_log(lines)
                return 14, DatosT

        # 15 -> Todos los usuarios en slack y no hay tramos
        if (DatosL.size == 0) and np.all(nod_u == slack):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 15\r\nTodos los usuarios están en el trafo y no hay tramos\r\n"])
            return 15, DatosT

        # 34 -> Todos los usuarios en slack y sí hay tramos
        if (DatosL.size != 0) and np.all(nod_u == slack):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 34\r\nTodos los usuarios están en el trafo y el circuito tiene tramos\r\n"])
            return 34, DatosT

        # 33 -> Hay usuarios fuera del slack y no hay tramos
        if (DatosL.size == 0) and np.any(nod_u != slack):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 33\r\nUsuarios conectados en nodos diferentes al trafo y el circuito no tiene tramos\r\n"])
            return 33, DatosT

        # 16 -> Fases de usuario válidas por tipo de trafo
        ok_mono_u = np.isin(fase_u, [1, 2, 4])
        ok_tri_u  = np.isin(fase_u, [1, 2, 3, 4, 5, 6, 7])
        if (tipo == 1 and not np.all(ok_mono_u)) or (tipo == 3 and not np.all(ok_tri_u)):
            lines = [f"\r\nCircuito: {circ}\r\nError: 16\r\nUsuarios con faseo incompatible con el trafo:\r\n"]
            idx_bad = np.where(~(ok_mono_u if tipo==1 else ok_tri_u))[0]
            for i in idx_bad:
                lines.append(f"{int(nod_u[i])}\r\n")
            _write_log(lines)
            return 16, DatosT

        # 18 -> Tipo de medidor ∈ {1,2}
        if not np.all(np.isin(med, [1, 2])):
            lines = [f"\r\nCircuito: {circ}\r\nError: 18\r\nUsuarios con tipo de medidor desconocido:\r\n"]
            for i in np.where(~np.isin(med, [1, 2]))[0]:
                lines.append(f"{int(nod_u[i])}\r\n")
            _write_log(lines)
            return 18, DatosT

        # 19 -> Estrato ∈ {0..6}
        if not np.all(np.isin(est, [0, 1, 2, 3, 4, 5, 6])):
            lines = [f"\r\nCircuito: {circ}\r\nError: 19\r\nUsuarios con estrato desconocido:\r\n"]
            for i in np.where(~np.isin(est, [0,1,2,3,4,5,6]))[0]:
                lines.append(f"{int(nod_u[i])}\r\n")
            _write_log(lines)
            return 19, DatosT

        # 20 -> Clase de servicio ∈ {1..11}
        if not np.all(np.isin(clas, np.arange(1, 12))):
            lines = [f"\r\nCircuito: {circ}\r\nError: 20\r\nUsuarios con clase de servicio desconocida:\r\n"]
            for i in np.where(~np.isin(clas, np.arange(1, 12)))[0]:
                lines.append(f"{int(nod_u[i])}\r\n")
            _write_log(lines)
            return 20, DatosT         
            
        # 27 -> Todos los usuarios conectados a la misma fase (solo reporto si es radial)
        if topo == 1 and DatosN.size:
            # excluir posibles NaN si los hubiera
            fvals = np.array(fase_u, dtype=float)
            fvals = fvals[~np.isnan(fvals)].astype(int)
            if fvals.size and np.unique(fvals).size == 1:
                f = int(np.unique(fvals)[0])
                # Etiquetas directas por código de fase de usuario:
                fase_txt = {
                    1: "la fase A",
                    2: "la fase B",
                    3: "la fase C",
                    4: "las fases A-B",
                    5: "las fases B-C",
                    6: "las fases C-A",
                    7: "las fases A-B-C",
                }.get(f, None)
        
                lines = [f"\r\nCircuito: {circ}\r\nError: 27\r\n"]
                if fase_txt is None:
                    lines.append("Distribución de fases no válida\r\n")
                else:
                    lines.append(f"Todos los usuarios están conectados a {fase_txt}\r\n")
                _write_log(lines)
                return 27, DatosT

    # =======================
    # GRAFO con NetworkX para (22, 24, 26) y para reglas adicionales
    # =======================
    
    if DatosL.size:
        edges = DatosL[:, :2].astype(int)
        phases = DatosL[:, 2].astype(int)

        # Construir grafo simple sin paralelos (para topología)
        G = nx.Graph()
        G.add_edges_from(map(tuple, np.sort(edges, axis=1)))  # no multiedges

        # 22 -> Islas (componentes conexas > 1)
        if nx.number_connected_components(G) > 1:
            _write_log([f"\r\nCircuito: {circ}\r\nError: 22\r\nEl circuito tiene islas\r\n"])
            return 22, DatosT

        # Conteo de nodos/aristas efectivas
        N = G.number_of_nodes()
        M = G.number_of_edges()

        # 26 -> Es radial (M = N-1) y viene marcado enmallado (topo=0)
        if (M == N - 1) and (topo == 0):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 26\r\nEl circuito es radial pero viene marcado como enmallado\r\n"])
            return 26, DatosT

        # 24 -> Tiene anillos (M >= N) y viene marcado radial (topo=1)
        if (M >= N) and (topo == 1):
            _write_log([f"\r\nCircuito: {circ}\r\nError: 24\r\nEl circuito es enmallado pero viene marcado como radial\r\n"])
            return 24, DatosT

        # =======================
        # Verificaciones de faseo según topología usando BFS/puentes
        # =======================

        # Preparar índice de faseo por arista (sin paralelos, normalizado por orden)
        # Usamos dict con llave (min(u,v), max(u,v)) -> fase
        phase_by_edge = {}
        for (u, v), f in zip(edges, phases):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            # Si hay paralelos, nos quedamos con el primer visto (o podrías validar que todos coinciden)
            if key not in phase_by_edge:
                phase_by_edge[key] = int(f)

        # --- RADIAL: (25) faseo en caminos consecutivos, (23) conexión de cargas vs tramo ---
        if topo == 1:
            # Orientar árbol por BFS desde slack (si slack no está, tomar un nodo cualquiera)
            root = slack if slack in G else next(iter(G.nodes))
            parent = {root: None}
            pedge = {}  # para cada nodo, arista con su padre
            from collections import deque
            dq = deque([root])

            while dq:
                u = dq.popleft()
                for v in G.neighbors(u):
                    if v in parent and parent[v] is not None and parent[v] == u:
                        continue
                    if v not in parent:
                        parent[v] = u
                        key = (min(u, v), max(u, v))
                        pedge[v] = key
                        dq.append(v)

            # (25) tramo→tramo: para cada nodo con padre y abuelo, comparar faseos
            bad_pairs = []
            for v, p in parent.items():
                if v == root or p is None:
                    continue
                gp = parent.get(p, None)
                if gp is None:
                    continue
                e1 = pedge[v]     # (p, v)
                e2 = pedge[p]     # (gp, p)
                f1 = phase_by_edge.get(e1, 0)
                f2 = phase_by_edge.get(e2, 0)
                if f1 == 0 or f2 == 0 or not COMP_TT[f2, f1]:
                    bad_pairs.append((e2[0], e2[1], e1[0], e1[1]))

            if bad_pairs:
                lines = [f"\r\nCircuito: {circ}\r\nError: 25\r\n"]
                for (a, b, c, d) in bad_pairs:
                    lines.append(f"Existe mal faseo de {a} - {b} a {c} - {d}\r\n")
                _write_log(lines)
                return 25, DatosT

            # (23) tramo→usuario: para cada nodo con usuarios, checar fase tramo entrante
            if DatosN.size:
                nod_u = DatosN[:, 0].astype(int)
                fase_u = DatosN[:, 1].astype(int)
                # agrupar usuarios por nodo
                from collections import defaultdict
                users_by_node = defaultdict(list)
                for n, fu in zip(nod_u, fase_u):
                    users_by_node[int(n)].append(int(fu))

                bad_nodes = []
                for v in users_by_node:
                    # fase del tramo que alimenta v (p→v). Para el root, no hay tramo entrante.
                    if v == root or v not in pedge:
                        # trafo monofásico: usuarios en slack deben estar en {1,2,4}
                        if v == root and tipo == 1:
                            if not set(users_by_node[v]).issubset({1, 2, 4}):
                                bad_nodes.append(v)
                        continue
                    ftramo = phase_by_edge.get(pedge[v], 0)
                    if ftramo == 0:
                        continue
                    # todos usuarios de v deben ser compatibles con ftramo
                    bad_local = [fu for fu in users_by_node[v] if not COMP_TU[ftramo, fu]]
                    if bad_local:
                        bad_nodes.append(v)

                if bad_nodes:
                    lines = [f"\r\nCircuito: {circ}\r\nError: 23\r\n"]
                    for n in bad_nodes:
                        lines.append(f"Hay una carga mal conectada en el nodo {n}\r\n")
                    _write_log(lines)
                    return 23, DatosT

        # --- ENMALLADO: (30) faseo inconsistente en secuencias locales, (31) cargas no alimentables ---
        else:
            # Idea: usar componentes biconexas para identificar "regiones de anillo".
            # (30) Para cada biconexa, probar triples (u–v–w) y verificar COMP_TT entre aristas contiguas.
            bad30 = []
            for bic in nx.biconnected_components(G):
                sub = G.subgraph(bic)
                # iterar sobre caminos de longitud 2 dentro de la biconexa
                for v in sub.nodes:
                    nbrs = list(sub.neighbors(v))
                    for i in range(len(nbrs)):
                        for j in range(i + 1, len(nbrs)):
                            a, b = nbrs[i], v
                            c = nbrs[j]
                            e1 = (min(a, b), max(a, b))
                            e2 = (min(b, c), max(b, c))
                            f1 = phase_by_edge.get(e1, 0)
                            f2 = phase_by_edge.get(e2, 0)
                            if f1 == 0 or f2 == 0:
                                continue
                            if not COMP_TT[f1, f2]:
                                bad30.append((e1[0], e1[1], e2[0], e2[1]))
            if bad30:
                lines = [f"\r\nCircuito: {circ}\r\nError: 30\r\nLos siguientes tramos de líneas tienen errores de faseo\r\n"]
                for (a, b, c, d) in bad30:
                    lines.append(f"De {a} - {b} a {c} {d}\r\n")
                _write_log(lines)
                return 30, DatosT

            # (31) Usuarios dentro de mallas: al menos un tramo incidente compatible
            if DatosN.size:
                nod_u = DatosN[:, 0].astype(int)
                fase_u = DatosN[:, 1].astype(int)
                # nodos que están en alguna biconexa de tamaño >= 3 (anillo)
                nodes_in_rings = set()
                for bic in nx.biconnected_components(G):
                    if len(bic) >= 3:
                        nodes_in_rings.update(bic)

                bad31 = []
                for node, fu in zip(nod_u, fase_u):
                    # tramos incidentes
                    ok = False
                    for v in G.neighbors(node) if node in G else []:
                        ekey = (min(node, v), max(node, v))
                        ft = phase_by_edge.get(ekey, 0)
                        if ft != 0 and COMP_TU[ft, int(fu)]:
                            ok = True
                            break
                    if not ok:
                        # si no hay tramo compatible que pueda alimentarlo en malla, marcarlo
                        bad31.append(int(node))

                if bad31:
                    lines = [f"\r\nCircuito: {circ}\r\nError: 31\r\n"]
                    for n in sorted(set(bad31)):
                        lines.append(f"Hay una carga mal conectada en el nodo {n}\r\n")
                    _write_log(lines)
                    return 31, DatosT

    # Si nada falló:
    _write_log([f"\r\nCircuito: {circ}\r\nError: 0\r\nCircuito normal\r\n"])
    return 0, DatosT
