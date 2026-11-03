import { useState } from "react";
import { pipeline, RawImage } from "@huggingface/transformers";

const MODELOS = ["Xenova/modnet", "studioludens/birefnet-lite-512"];

const TIPOS = [
  "camiseta",
  "camisa",
  "pantalón",
  "pantalón corto",
  "falda",
  "vestido",
  "chaqueta",
  "abrigo",
  "sudadera",
  "jersey",
  "zapatos",
  "bolso",
];
const COLORES = [
  "negro",
  "blanco",
  "gris",
  "beige",
  "marrón",
  "azul",
  "azul marino",
  "verde",
  "rojo",
  "rosa",
  "morado",
  "amarillo",
  "naranja",
];
const PATRONES = [
  "liso",
  "rayas",
  "cuadros",
  "estampado floral",
  "estampado geométrico",
  "degradado",
  "estampado abstracto",
];

let clasificadorCacheado: any = null;

async function getClasificador() {
  if (!clasificadorCacheado) {
    const opciones: { device?: "webgpu" } = navigator.gpu
      ? { device: "webgpu" }
      : {};
    clasificadorCacheado = await pipeline(
      "zero-shot-image-classification",
      "ff13/fashion-clip",
      opciones,
    );
  }
  return clasificadorCacheado;
}

type Resultado = {
  modelo: string;
  imagen: string;
  tiempo: number;
  puntuacion: number;
};

type Tags = {
  tipo: string;
  color: string;
  patron: string;
};

function calcularMetricas(img: RawImage, fotoOriginal: HTMLImageElement) {
  const w = img.width;
  const h = img.height;
  const data = img.data as Uint8ClampedArray;

  const canvasOrig = document.createElement("canvas");
  canvasOrig.width = w;
  canvasOrig.height = h;
  const ctxOrig = canvasOrig.getContext("2d")!;
  ctxOrig.drawImage(fotoOriginal, 0, 0, w, h);
  const origData = ctxOrig.getImageData(0, 0, w, h).data;

  let sumaGradOrig = 0;
  const gradOrig: number[] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const gray =
        0.299 * origData[i] + 0.587 * origData[i + 1] + 0.114 * origData[i + 2];
      const grayR =
        0.299 * origData[i + 4] +
        0.587 * origData[i + 5] +
        0.114 * origData[i + 6];
      const grayD =
        0.299 * origData[i + w * 4] +
        0.587 * origData[i + w * 4 + 1] +
        0.114 * origData[i + w * 4 + 2];
      const grad = Math.abs(grayR - gray) + Math.abs(grayD - gray);
      gradOrig.push(grad);
      sumaGradOrig += grad;
    }
  }

  let sumaGradAlpha = 0;
  let coincidencia = 0;
  let idx = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const a = data[i + 3];
      const aR = data[i + 4 + 3];
      const aD = data[i + w * 4 + 3];
      const gradAlpha = Math.abs(aR - a) + Math.abs(aD - a);
      sumaGradAlpha += gradAlpha;
      if (gradOrig[idx] > 30 && gradAlpha > 30) coincidencia++;
      idx++;
    }
  }

  const totalPix = (w - 2) * (h - 2);
  const preservacionBordes = sumaGradOrig > 0 ? coincidencia / totalPix : 0.5;
  const densidadBordes = sumaGradAlpha / totalPix;
  const completitud =
    Array.from(data).filter((_, i) => i % 4 === 3 && data[i] > 128).length /
    (w * h);
  const penalizacion = completitud < 0.1 || completitud > 0.9 ? 0.5 : 1;

  return { preservacionBordes, densidadBordes, penalizacion };
}

async function ejecutarModelo(
  modelo: string,
  foto: string,
  fotoOriginal: HTMLImageElement,
): Promise<Resultado> {
  const inicio = performance.now();
  const quitarFondo = await pipeline("background-removal", modelo);
  const salida = await quitarFondo(foto);
  const canvas = document.createElement("canvas");
  canvas.width = salida.width;
  canvas.height = salida.height;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(salida.toCanvas(), 0, 0);
  const imagen = canvas.toDataURL("image/png");
  const tiempo = (performance.now() - inicio) / 1000;
  const m = calcularMetricas(salida, fotoOriginal);
  const puntuacion =
    m.preservacionBordes * 0.6 + m.densidadBordes * 0.2 + m.penalizacion * 0.2;
  return { modelo, imagen, tiempo, puntuacion };
}

async function generarTags(imagenRecortada: string): Promise<Tags> {
  const clasificador = await getClasificador();
  const img = await RawImage.fromURL(imagenRecortada);

  const [tipoRes, colorRes, patronRes] = await Promise.all([
    clasificador(img, TIPOS),
    clasificador(img, COLORES),
    clasificador(img, PATRONES),
  ]);

  return {
    tipo: tipoRes[0].label,
    color: colorRes[0].label,
    patron: patronRes[0].label,
  };
}

export default function App() {
  const [foto, setFoto] = useState<string | null>(null);
  const [resultado, setResultado] = useState<Resultado | null>(null);
  const [tags, setTags] = useState<Tags | null>(null);
  const [cargando, setCargando] = useState(false);
  const [detalle, setDetalle] = useState<string>("");

  async function recortar() {
    if (!foto) return;
    setCargando(true);
    setDetalle("");
    setTags(null);
    try {
      const imgOrig = new Image();
      imgOrig.src = foto;
      await new Promise((resolve) => {
        imgOrig.onload = resolve;
      });
      const resultados = await Promise.all(
        MODELOS.map((m) => ejecutarModelo(m, foto, imgOrig)),
      );
      const mejor = resultados.reduce((a, b) =>
        b.puntuacion > a.puntuacion ? b : a,
      );
      setResultado(mejor);
      setDetalle(
        `Ganador: ${mejor.modelo} (puntuación ${mejor.puntuacion.toFixed(3)}, ${mejor.tiempo.toFixed(1)}s)`,
      );
      const tagsGenerados = await generarTags(mejor.imagen);
      setTags(tagsGenerados);
    } catch (e) {
      alert("Error: " + (e as Error).message);
    }
    setCargando(false);
  }

  return (
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h1>Laboratorio de recorte</h1>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFoto(URL.createObjectURL(e.target.files![0]))}
      />
      <button onClick={recortar} disabled={!foto || cargando}>
        {cargando ? "Procesando..." : "Recortar e identificar"}
      </button>
      {detalle && <p>{detalle}</p>}
      {tags && (
        <div
          style={{
            marginTop: 10,
            padding: 10,
            background: "#f0f0f0",
            borderRadius: 8,
          }}
        >
          <strong>Tags detectados:</strong>
          <br />
          Tipo: {tags.tipo}
          <br />
          Color: {tags.color}
          <br />
          Patrón: {tags.patron}
        </div>
      )}
      <div style={{ display: "flex", gap: 20, marginTop: 20 }}>
        {foto && <img src={foto} width={300} alt="Original" />}
        {resultado && (
          <img src={resultado.imagen} width={300} alt="Recortada" />
        )}
      </div>
    </div>
  );
}
