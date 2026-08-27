// Перематирование PNG с белой подложки на заданный цвет фона.
// Картинки экспортированы поверх белого: пиксели сглаживания сохранились
// непрозрачными и почти белыми. На небелом фоне они читаются как ореол.
import fs from "fs";
import zlib from "zlib";

const CRC = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) { let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c; }
  return (buf) => { let c = -1;
    for (let i = 0; i < buf.length; i++) c = t[(c ^ buf[i]) & 255] ^ (c >>> 8);
    return (c ^ -1) >>> 0; };
})();

function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(CRC(body));
  return Buffer.concat([len, body, crc]);
}

function decode(file) {
  const b = fs.readFileSync(file);
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  if (b[24] !== 8 || b[25] !== 6) throw new Error("ожидался 8-битный RGBA");
  const idat = [];
  for (let i = 8; i < b.length - 8; ) {
    const len = b.readUInt32BE(i), typ = b.toString("ascii", i + 4, i + 8);
    if (typ === "IDAT") idat.push(b.slice(i + 8, i + 8 + len));
    if (typ === "IEND") break;
    i += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const bpp = 4, stride = w * bpp, px = Buffer.alloc(h * stride);
  let p = 0;
  for (let y = 0; y < h; y++) {
    const ft = raw[p++], line = raw.slice(p, p + stride); p += stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? px[y * stride + x - bpp] : 0;
      const bb = y > 0 ? px[(y - 1) * stride + x] : 0;
      const c = (x >= bpp && y > 0) ? px[(y - 1) * stride + x - bpp] : 0;
      let v = line[x];
      if (ft === 1) v += a; else if (ft === 2) v += bb;
      else if (ft === 3) v += (a + bb) >> 1;
      else if (ft === 4) {
        const q = a + bb - c, pa = Math.abs(q - a), pb = Math.abs(q - bb), pc = Math.abs(q - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? bb : c);
      }
      px[y * stride + x] = v & 255;
    }
  }
  return { w, h, px };
}

function encode(w, h, px) {
  const stride = w * 4, raw = Buffer.alloc(h * (stride + 1));
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0;                                  // фильтр None
    px.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6;                                     // 8 бит, RGBA
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const [file, hex, outFile] = process.argv.slice(2);
const B = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const { w, h, px } = decode(file);

// Чем ближе непрозрачный пиксель к белому, тем сильнее тянем его к цвету фона:
// t=0 на пороге (цвет остаётся как был), t=1 на чистом белом (становится фоном).
const T = 200;
let touched = 0;
for (let i = 0; i < px.length; i += 4) {
  if (px[i + 3] < 8) continue;
  const lo = Math.min(px[i], px[i + 1], px[i + 2]);
  if (lo <= T) continue;
  const t = (lo - T) / (255 - T);
  for (let k = 0; k < 3; k++) px[i + k] = Math.round(px[i + k] * (1 - t) + B[k] * t);
  touched++;
}
fs.writeFileSync(outFile, encode(w, h, px));
console.log(`${file.split("/").pop()} -> ${outFile.split("/").pop()}: перекрашено ${touched} px, ${(fs.statSync(outFile).size / 1024).toFixed(1)} KB`);
