"""Convert MediaPipe's face-mesh TFLite model to ONNX - with no extra packages.

MediaPipe's runtime needs AVX, but its *models* are plain data: a TFLite file
is a FlatBuffer holding the graph and the weights. This module reads that
FlatBuffer with a few lines of ``struct`` code, translates the handful of
operators the face models use into ONNX nodes (NHWC -> NCHW), folds the
float16 weights into float32 initialisers and serialises the ONNX protobuf by
hand. ONNX Runtime selects CPU kernels at run time, so the converted model runs
on processors without AVX as well.

The output is deterministic (same input bytes -> same output bytes), which lets
the model registry pin its checksum.

Supported operators: CONV_2D, DEPTHWISE_CONV_2D, PRELU, ADD, MUL, MAX_POOL_2D,
AVERAGE_POOL_2D, PAD, LOGISTIC, RELU, RELU6, TANH, RESHAPE, CONCATENATION,
DEQUANTIZE (float16 weights).
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- FlatBuffer reader


class _FB:
    """Minimal FlatBuffer table access (little endian, as TFLite writes it)."""

    def __init__(self, buf: bytes):
        self.b = buf

    def u8(self, p):
        return self.b[p]

    def i8(self, p):
        return struct.unpack_from("<b", self.b, p)[0]

    def u16(self, p):
        return struct.unpack_from("<H", self.b, p)[0]

    def i32(self, p):
        return struct.unpack_from("<i", self.b, p)[0]

    def u32(self, p):
        return struct.unpack_from("<I", self.b, p)[0]

    def u64(self, p):
        return struct.unpack_from("<Q", self.b, p)[0]

    def root(self) -> int:
        return self.u32(0)

    def field(self, t: int, slot: int) -> int:
        """Absolute position of field ``slot`` of the table at ``t`` (0 if absent)."""
        vt = t - self.i32(t)
        vsize = self.u16(vt)
        o = 4 + 2 * slot
        if o >= vsize:
            return 0
        off = self.u16(vt + o)
        return t + off if off else 0

    def deref(self, p: int) -> int:
        return p + self.u32(p)

    def scalar(self, t, slot, fmt, default=0):
        p = self.field(t, slot)
        return struct.unpack_from("<" + fmt, self.b, p)[0] if p else default

    def table(self, t, slot):
        p = self.field(t, slot)
        return self.deref(p) if p else 0

    def vector(self, t, slot):
        """(start, length) of a vector field, or (0, 0)."""
        p = self.field(t, slot)
        if not p:
            return 0, 0
        v = self.deref(p)
        return v + 4, self.u32(v)

    def tables(self, t, slot) -> list[int]:
        s, n = self.vector(t, slot)
        return [self.deref(s + 4 * i) for i in range(n)]

    def ints(self, t, slot, fmt="i") -> list[int]:
        s, n = self.vector(t, slot)
        if not n:
            return []
        return list(struct.unpack_from(f"<{n}{fmt}", self.b, s))

    def string(self, t, slot) -> str:
        p = self.field(t, slot)
        if not p:
            return ""
        s = self.deref(p)
        return self.b[s + 4 : s + 4 + self.u32(s)].decode("utf-8", "replace")

    def bytes_(self, t, slot) -> bytes:
        s, n = self.vector(t, slot)
        return self.b[s : s + n]


# TFLite enums (schema.fbs)
_OPS = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D",
    6: "DEQUANTIZE", 14: "LOGISTIC", 17: "MAX_POOL_2D", 18: "MUL", 19: "RELU", 21: "RELU6",
    22: "RESHAPE", 28: "TANH", 34: "PAD", 54: "PRELU",
}  # fmt: skip
_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 9: np.int8, 10: np.float64}


class TFLiteModel:
    """The parts of a TFLite model needed for conversion."""

    def __init__(self, data: bytes):
        fb = _FB(data)
        m = fb.root()
        codes = []
        for oc in fb.tables(m, 1):
            dep = fb.scalar(oc, 0, "b", 0)
            code = fb.scalar(oc, 3, "i", 0)
            codes.append(max(dep, code))
        self.buffers = []
        for b in fb.tables(m, 4):
            off, size = fb.scalar(b, 1, "Q", 0), fb.scalar(b, 2, "Q", 0)
            self.buffers.append(data[off : off + size] if off > 1 else fb.bytes_(b, 0))
        sg = fb.tables(m, 2)
        if len(sg) != 1:
            raise ValueError(f"expected one subgraph, found {len(sg)}")
        g = sg[0]
        self.tensors = []
        for t in fb.tables(g, 0):
            q = fb.table(t, 4)
            quantized = bool(q) and fb.vector(q, 2)[1] > 0  # QuantizationParameters.scale
            self.tensors.append(
                {
                    "shape": fb.ints(t, 0),
                    "type": fb.scalar(t, 1, "b", 0),
                    "buffer": fb.scalar(t, 2, "I", 0),
                    "name": fb.string(t, 3),
                    "quantized": quantized,
                }
            )
        self.inputs = fb.ints(g, 1)
        self.outputs = fb.ints(g, 2)
        self.ops = []
        for o in fb.tables(g, 3):
            code = codes[fb.scalar(o, 0, "I", 0)]
            opt = fb.table(o, 4)
            self.ops.append({"op": _OPS.get(code, f"OP{code}"), "inputs": fb.ints(o, 1), "outputs": fb.ints(o, 2), "opt": opt})
        self._fb = fb

    def const(self, ti: int) -> np.ndarray | None:
        t = self.tensors[ti]
        raw = self.buffers[t["buffer"]] if t["buffer"] < len(self.buffers) else b""
        if not raw:
            return None
        if t["quantized"]:
            raise ValueError(f"quantized tensor {t['name']!r} is not supported")
        dt = _DTYPES.get(t["type"])
        if dt is None:
            raise ValueError(f"unsupported tensor type {t['type']} for {t['name']!r}")
        return np.frombuffer(raw, dtype=np.dtype(dt).newbyteorder("<")).reshape(t["shape"]).copy()

    def options(self, op: dict, layout: list[tuple[int, str, int]]) -> dict:
        """Read builtin options: ``layout`` = [(slot, struct fmt, default), ...]."""
        fb, t = self._fb, op["opt"]
        return {i: (fb.scalar(t, slot, fmt, d) if t else d) for i, (slot, fmt, d) in enumerate(layout)}


# --------------------------------------------------------------------------- ONNX protobuf writer


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _ld(field: int, payload: bytes) -> bytes:  # length-delimited
    return _key(field, 2) + _varint(len(payload)) + payload


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(int(v))


def _str(field: int, s: str) -> bytes:
    return _ld(field, s.encode("utf-8"))


_ONNX_FLOAT, _ONNX_INT64 = 1, 7


def _tensor(name: str, arr: np.ndarray) -> bytes:
    if arr.dtype == np.float32:
        dt = _ONNX_FLOAT
    elif arr.dtype == np.int64:
        dt = _ONNX_INT64
    else:
        raise TypeError(arr.dtype)
    body = b"".join(_int(1, d) for d in arr.shape) + _int(2, dt) + _str(8, name)
    return body + _ld(9, np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes())


def _attr(name: str, v) -> bytes:
    body = _str(1, name)
    if isinstance(v, float):
        return body + _key(2, 5) + struct.pack("<f", v) + _int(20, 1)
    if isinstance(v, int):
        return body + _int(3, v) + _int(20, 2)
    if isinstance(v, (list, tuple)):
        return body + b"".join(_int(8, int(x)) for x in v) + _int(20, 7)
    raise TypeError(type(v))


def _node(op: str, inputs: list[str], outputs: list[str], name: str, **attrs) -> bytes:
    body = b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs) + _str(3, name) + _str(4, op)
    return body + b"".join(_ld(5, _attr(k, attrs[k])) for k in sorted(attrs))


def _value_info(name: str, shape: list[int]) -> bytes:
    dims = b"".join(_ld(1, _int(1, d)) for d in shape)
    tensor_type = _int(1, _ONNX_FLOAT) + _ld(2, dims)
    return _str(1, name) + _ld(2, _ld(1, tensor_type))


def _model(nodes: list[bytes], inits: list[bytes], inputs: list[bytes], outputs: list[bytes], opset: int, doc: str) -> bytes:
    graph = b"".join(_ld(1, n) for n in nodes) + _str(2, "graph") + b"".join(_ld(5, t) for t in inits)
    graph += b"".join(_ld(11, i) for i in inputs) + b"".join(_ld(12, o) for o in outputs)
    return _int(1, 8) + _str(2, "lumen-tflite2onnx") + _str(3, "1") + _str(6, doc) + _ld(7, graph) + _ld(8, _str(1, "") + _int(2, opset))


# --------------------------------------------------------------------------- conversion


def _same_pads(in_size: int, k: int, stride: int, dil: int) -> tuple[int, int]:
    """TensorFlow 'SAME' padding (the extra pixel goes after)."""
    eff = (k - 1) * dil + 1
    out = -(-in_size // stride)
    total = max((out - 1) * stride + eff - in_size, 0)
    return total // 2, total - total // 2


def convert(data: bytes, doc: str = "") -> bytes:
    """TFLite model bytes -> ONNX model bytes (input NHWC float32, outputs in TFLite layout)."""
    m = TFLiteModel(data)
    T = m.tensors
    nodes: list[bytes] = []
    inits: list[bytes] = []
    const_of: dict[int, np.ndarray] = {}  # outputs of DEQUANTIZE (folded weights)
    names: dict[int, str] = {}
    counter = [0]

    def uniq(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def const(ti: int) -> np.ndarray | None:
        if ti in const_of:
            return const_of[ti]
        return m.const(ti)

    def init(arr: np.ndarray, prefix: str = "w") -> str:
        n = uniq(prefix)
        inits.append(_tensor(n, arr))
        return n

    def act(ti_out: int, fused: int, src: str) -> None:
        dst = f"t{ti_out}"
        if fused == 0:
            if src != dst:
                nodes.append(_node("Identity", [src], [dst], uniq("id")))
        elif fused == 1:
            nodes.append(_node("Relu", [src], [dst], uniq("relu")))
        elif fused == 3:
            lo, hi = init(np.array(0.0, np.float32), "c"), init(np.array(6.0, np.float32), "c")
            nodes.append(_node("Clip", [src, lo, hi], [dst], uniq("relu6")))
        elif fused == 2:
            lo, hi = init(np.array(-1.0, np.float32), "c"), init(np.array(1.0, np.float32), "c")
            nodes.append(_node("Clip", [src, lo, hi], [dst], uniq("relu1")))
        elif fused == 4:
            nodes.append(_node("Tanh", [src], [dst], uniq("tanh")))
        else:
            raise ValueError(f"unsupported fused activation {fused}")
        names[ti_out] = dst

    def operand(ti: int) -> str:
        """Name of an activation (NCHW) or a constant broadcastable in NCHW."""
        if ti in names:
            return names[ti]
        c = const(ti)
        if c is None:
            raise ValueError(f"tensor {T[ti]['name']!r} is used before it is produced")
        c = c.astype(np.float32)
        if c.ndim == 4:
            c = c.transpose(0, 3, 1, 2)
        elif c.ndim == 3:
            c = c.transpose(2, 0, 1)
        return init(c, "k")

    graph_in = []
    for ti in m.inputs:
        shp = T[ti]["shape"]
        graph_in.append(_value_info("input", shp))
        if len(shp) == 4:
            nodes.append(_node("Transpose", ["input"], [f"t{ti}"], uniq("to_nchw"), perm=[0, 3, 1, 2]))
            names[ti] = f"t{ti}"
        else:
            names[ti] = "input"

    for op in m.ops:
        kind, ins, outs = op["op"], op["inputs"], op["outputs"]
        if kind == "DEQUANTIZE":
            c = m.const(ins[0])
            if c is None:
                raise ValueError("DEQUANTIZE of a non-constant tensor is not supported")
            const_of[outs[0]] = c.astype(np.float32)
            continue
        ishape = T[ins[0]]["shape"]
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            w = const(ins[1]).astype(np.float32)
            if kind == "CONV_2D":
                o = m.options(op, [(0, "b", 0), (1, "i", 1), (2, "i", 1), (3, "b", 0), (4, "i", 1), (5, "i", 1)])
                pad, sw, sh, fused, dw, dh = (o[i] for i in range(6))
                wk = w.transpose(0, 3, 1, 2)  # OHWI -> OIHW
                group = 1
            else:
                o = m.options(op, [(0, "b", 0), (1, "i", 1), (2, "i", 1), (3, "i", 1), (4, "b", 0), (5, "i", 1), (6, "i", 1)])
                pad, sw, sh, _mult, fused, dw, dh = (o[i] for i in range(7))
                wk = w.transpose(3, 0, 1, 2)  # 1HW(C*M) -> (C*M)1HW
                group = ishape[3]
            kh, kw = wk.shape[2], wk.shape[3]
            if pad == 0:
                pt, pb = _same_pads(ishape[1], kh, sh, dh)
                pl, pr = _same_pads(ishape[2], kw, sw, dw)
            else:
                pt = pb = pl = pr = 0
            args = [names[ins[0]], init(np.ascontiguousarray(wk), "w")]
            if len(ins) > 2 and ins[2] >= 0:
                args.append(init(const(ins[2]).astype(np.float32).reshape(-1), "b"))
            tmp = uniq("conv")
            nodes.append(_node("Conv", args, [tmp], tmp, kernel_shape=[kh, kw], strides=[sh, sw], dilations=[dh, dw], pads=[pt, pl, pb, pr], group=group))
            act(outs[0], fused, tmp)
        elif kind in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            o = m.options(op, [(0, "b", 0), (1, "i", 1), (2, "i", 1), (3, "i", 1), (4, "i", 1), (5, "b", 0)])
            pad, sw, sh, fw, fh, fused = (o[i] for i in range(6))
            if pad == 0:
                pt, pb = _same_pads(ishape[1], fh, sh, 1)
                pl, pr = _same_pads(ishape[2], fw, sw, 1)
            else:
                pt = pb = pl = pr = 0
            tmp = uniq("pool")
            extra = {"count_include_pad": 0} if kind == "AVERAGE_POOL_2D" else {}
            nodes.append(_node("MaxPool" if kind == "MAX_POOL_2D" else "AveragePool", [names[ins[0]]], [tmp], tmp, kernel_shape=[fh, fw], strides=[sh, sw], pads=[pt, pl, pb, pr], **extra))
            act(outs[0], fused, tmp)
        elif kind == "PRELU":
            a = const(ins[1]).astype(np.float32)
            a = a.reshape((1,) * (3 - a.ndim) + a.shape) if a.ndim < 3 else a.reshape(a.shape[-3:])
            nodes.append(_node("PRelu", [names[ins[0]], init(np.ascontiguousarray(a.transpose(2, 0, 1)), "alpha")], [f"t{outs[0]}"], uniq("prelu")))
            names[outs[0]] = f"t{outs[0]}"
        elif kind in ("ADD", "MUL"):
            o = m.options(op, [(0, "b", 0)])
            tmp = uniq(kind.lower())
            nodes.append(_node("Add" if kind == "ADD" else "Mul", [operand(ins[0]), operand(ins[1])], [tmp], tmp))
            act(outs[0], o[0], tmp)
        elif kind == "PAD":
            p = const(ins[1]).astype(np.int64).reshape(-1, 2)
            if p.shape[0] != 4:
                raise ValueError("PAD is only supported on 4D tensors")
            p = p[[0, 3, 1, 2]]  # NHWC -> NCHW
            pads = np.concatenate([p[:, 0], p[:, 1]]).astype(np.int64)
            nodes.append(_node("Pad", [names[ins[0]], init(pads, "pads")], [f"t{outs[0]}"], uniq("pad")))  # constant 0
            names[outs[0]] = f"t{outs[0]}"
        elif kind in ("LOGISTIC", "RELU", "RELU6", "TANH"):
            if kind == "LOGISTIC":
                nodes.append(_node("Sigmoid", [names[ins[0]]], [f"t{outs[0]}"], uniq("sigmoid")))
                names[outs[0]] = f"t{outs[0]}"
            else:
                act(outs[0], {"RELU": 1, "RELU6": 3, "TANH": 4}[kind], names[ins[0]])
        elif kind == "CONCATENATION":
            o = m.options(op, [(0, "i", 0), (1, "b", 0)])
            axis = o[0] % len(ishape)
            axis = [0, 2, 3, 1][axis] if len(ishape) == 4 else axis
            tmp = uniq("concat")
            nodes.append(_node("Concat", [operand(i) for i in ins], [tmp], tmp, axis=axis))
            act(outs[0], o[1], tmp)
        elif kind == "RESHAPE":
            src = names[ins[0]]
            if len(ishape) == 4:  # back to NHWC element order before reshaping
                t2 = uniq("to_nhwc")
                nodes.append(_node("Transpose", [src], [t2], t2, perm=[0, 2, 3, 1]))
                src = t2
            oshape = T[outs[0]]["shape"]
            dst = f"t{outs[0]}"
            if len(oshape) == 4:
                t3 = uniq("reshape")
                nodes.append(_node("Reshape", [src, init(np.array(oshape, np.int64), "shape")], [t3], t3))
                nodes.append(_node("Transpose", [t3], [dst], uniq("to_nchw"), perm=[0, 3, 1, 2]))
            else:
                nodes.append(_node("Reshape", [src, init(np.array(oshape, np.int64), "shape")], [dst], uniq("reshape")))
                names[outs[0]] = dst
                T[outs[0]]["_flat"] = True
                continue
            names[outs[0]] = dst
        else:
            raise ValueError(f"TFLite operator {kind} is not supported by this converter")

    graph_out = []
    for ti in m.outputs:
        shp = T[ti]["shape"]
        oname = T[ti]["name"].replace(":", "_") or f"output{ti}"
        if len(shp) == 4 and not T[ti].get("_flat"):
            nodes.append(_node("Transpose", [names[ti]], [oname], uniq("out"), perm=[0, 2, 3, 1]))
        else:
            nodes.append(_node("Identity", [names[ti]], [oname], uniq("out")))
        graph_out.append(_value_info(oname, shp))
    return _model(nodes, inits, graph_in, graph_out, opset=13, doc=doc)


def _pb_fields(b: bytes):
    """Decode one protobuf message level: yields (field, wire_type, value)."""

    def varint(p):
        r = s = 0
        while True:
            c = b[p]
            p += 1
            r |= (c & 0x7F) << s
            s += 7
            if not c & 0x80:
                return r, p

    p = 0
    while p < len(b):
        k, p = varint(p)
        f, w = k >> 3, k & 7
        if w == 0:
            v, p = varint(p)
        elif w == 1:
            v, p = b[p : p + 8], p + 8
        elif w == 5:
            v, p = b[p : p + 4], p + 4
        elif w == 2:
            n, p = varint(p)
            v, p = b[p : p + n], p + n
        else:
            raise ValueError(f"unsupported protobuf wire type {w}")
        yield f, w, v


def read_geometry_metadata(data: bytes) -> dict:
    """MediaPipe GeometryPipelineMetadata (canonical face mesh + Procrustes weights)."""

    def floats(fw):
        f, w, v = fw
        return [struct.unpack("<f", v)[0]] if w == 5 else list(struct.unpack(f"<{len(v) // 4}f", v))

    def uints(fw):
        f, w, v = fw
        return [v] if w == 0 else _packed_varints(v)

    mesh, basis = None, []
    for f, w, v in _pb_fields(data):
        if f == 1 and w == 2:
            mesh = v
        elif f == 2 and w == 2:
            d = {ff: vv for ff, _, vv in _pb_fields(v)}
            basis.append((int(d.get(1, 0)), struct.unpack("<f", d[2])[0] if 2 in d else 0.0))
    if mesh is None:
        raise ValueError("no canonical mesh in geometry metadata")
    vb, ib = [], []
    for fw in _pb_fields(mesh):
        if fw[0] == 3:
            vb += floats(fw)
        elif fw[0] == 4:
            ib += uints(fw)
    return {
        "vertices": np.array(vb, np.float32).reshape(-1, 5),  # x, y, z (cm), u, v
        "triangles": np.array(ib, np.int32).reshape(-1, 3),
        "basis_ids": np.array([i for i, _ in basis], np.int32),
        "basis_w": np.array([w for _, w in basis], np.float32),
    }


def _packed_varints(v: bytes) -> list[int]:
    out, r, s = [], 0, 0
    for c in v:
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            out.append(r)
            r = s = 0
    return out


def read_task_member(task_path: str | Path, member: str) -> bytes:
    """Read one model file from a MediaPipe ``.task`` bundle (a zip archive)."""
    with zipfile.ZipFile(task_path) as z:
        return z.read(member)


def convert_face_mesh(task_path: str | Path, out_path: str | Path) -> Path:
    """face_landmarker.task -> ONNX face-mesh landmark model (atomic write)."""
    data = read_task_member(task_path, "face_landmarks_detector.tflite")
    onnx_bytes = convert(data, doc="MediaPipe Face Mesh V2 landmark model (Apache-2.0), converted from face_landmarker.task by Lumen Studio")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(onnx_bytes)
    tmp.replace(out)
    return out
